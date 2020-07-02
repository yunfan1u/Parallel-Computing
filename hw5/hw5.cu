#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cfloat>

using namespace std;

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
__host__ __device__ double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

__device__ __managed__ int n, planet, asteroid;  // global


void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

__global__ void run_step_kernel(int step, double* qx, double* qy, double* qz, double* vx, double* vy, double* vz, double* m, int* type) {
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < n) {
        double ax = 0;
        double ay = 0;
        double az = 0;
        
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            if (type[j] == 1) {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }
            //__syncthreads();
            
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 = pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax += param::G * mj * dx / dist3;
            ay += param::G * mj * dy / dist3;
            az += param::G * mj * dz / dist3;
            //__syncthreads();
        }
        
        //__syncthreads();
        vx[i] += ax * param::dt;
        vy[i] += ay * param::dt;
        vz[i] += az * param::dt;

        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
    

}



double dist(double x1, double y1, double z1, double x2, double y2, double z2)
{
    double dx = x1 - x2;
    double dy = y1 - y2;
    double dz = z1 - z2;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

double project_x(double x1, double y1, double z1, double x2, double y2, double z2)
{
    double dx = x1 - x2;
    double dy = y1 - y2;
    double dz = z1 - z2;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    double dist_xy = sqrt(dx*dx + dy*dy);
    
    return (dist_xy/dist) * (fabs(x1-x2)/dist_xy);
}

double project_y(double x1, double y1, double z1, double x2, double y2, double z2)
{
    double dx = x1 - x2;
    double dy = y1 - y2;
    double dz = z1 - z2;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    double dist_xy = sqrt(dx*dx + dy*dy);
    
    return (dist_xy/dist) * (fabs(y1-y2)/dist_xy);
}

double project_z(double x1, double y1, double z1, double x2, double y2, double z2)
{
    double dx = x1 - x2;
    double dy = y1 - y2;
    double dz = z1 - z2;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    
    return (fabs(z1-z2)/dist);
}


int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    
    //int n;  // n: n bodies
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;
    vector<int> t_temp;

    auto distance = [&](int i, int j) -> double {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };

    
    // read in
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    
    int t[n];
    for(int i=0; i<n; i++){
        if(type[i] == "device")
            t[i] = 1;
        else
            t[i] = 0;
    }
    
    int threadsPerBlock = 32;
    int blocksPerGrid = (n + threadsPerBlock-1)/threadsPerBlock;
    
    
    
    
    
    
    // convert std::vector to array
    // for P1
    double qx_h1[n], qy_h1[n], qz_h1[n];
    copy(qx.begin(), qx.end(), qx_h1);
    copy(qy.begin(), qy.end(), qy_h1);
    copy(qz.begin(), qz.end(), qz_h1);
    
    double vx_h1[n], vy_h1[n], vz_h1[n];
    copy(vx.begin(), vx.end(), vx_h1);
    copy(vy.begin(), vy.end(), vy_h1);
    copy(vz.begin(), vz.end(), vz_h1);
    
    double m_h1[n];
    copy(m.begin(), m.end(), m_h1);
    
    vector<int> devList;  // device list for P3
    for (int i = 0; i < n; i++) {
        if (t[i] == 1) {
            devList.push_back(i);
            m_h1[i] = 0;    // no device in problem 1
        }
    }
    
    // for P2
    double qx_h2[n], qy_h2[n], qz_h2[n];
    copy(qx.begin(), qx.end(), qx_h2);
    copy(qy.begin(), qy.end(), qy_h2);
    copy(qz.begin(), qz.end(), qz_h2);
    
    double vx_h2[n], vy_h2[n], vz_h2[n];
    copy(vx.begin(), vx.end(), vx_h2);
    copy(vy.begin(), vy.end(), vy_h2);
    copy(vz.begin(), vz.end(), vz_h2);
    
    double m_h2[n];
    copy(m.begin(), m.end(), m_h2);
    
    
    // device's variable
    double *qx_dev1, *qy_dev1, *qz_dev1, *vx_dev1, *vy_dev1, *vz_dev1, *m_dev1;
    int *t_dev1;
    
    double *qx_dev2, *qy_dev2, *qz_dev2, *vx_dev2, *vy_dev2, *vz_dev2, *m_dev2;
    int *t_dev2;
    
    
    // Problem 1 & 2
    double min_dist = std::numeric_limits<double>::infinity();
    int hit_time_step = -2;
    

    cudaSetDevice(0);
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    cudaMalloc(&qx_dev1, sizeof(double)*n);
    cudaMemcpyAsync(qx_dev1, qx_h1, sizeof(double)*n, cudaMemcpyHostToDevice, stream1);
    cudaMalloc(&qy_dev1, sizeof(double)*n);
    cudaMemcpyAsync(qy_dev1, qy_h1, sizeof(double)*n, cudaMemcpyHostToDevice, stream1);
    cudaMalloc(&qz_dev1, sizeof(double)*n);
    cudaMemcpyAsync(qz_dev1, qz_h1, sizeof(double)*n, cudaMemcpyHostToDevice, stream1);
    
    cudaMalloc(&vx_dev1, sizeof(double)*n);
    cudaMemcpyAsync(vx_dev1, vx_h1, sizeof(double)*n, cudaMemcpyHostToDevice, stream1);
    cudaMalloc(&vy_dev1, sizeof(double)*n);
    cudaMemcpyAsync(vy_dev1, vy_h1, sizeof(double)*n, cudaMemcpyHostToDevice, stream1);
    cudaMalloc(&vz_dev1, sizeof(double)*n);
    cudaMemcpyAsync(vz_dev1, vz_h1, sizeof(double)*n, cudaMemcpyHostToDevice, stream1);
    
    cudaMalloc(&m_dev1, sizeof(double)*n);
    cudaMemcpyAsync(m_dev1, m_h1, sizeof(double)*n, cudaMemcpyHostToDevice, stream1);
    
    cudaMalloc(&t_dev1, sizeof(int)*n);
    cudaMemcpyAsync(t_dev1, t, sizeof(int)*n, cudaMemcpyHostToDevice, stream1);
    
    
    cudaMalloc(&qx_dev2, sizeof(double)*n);
    cudaMemcpyAsync(qx_dev2, qx_h2, sizeof(double)*n, cudaMemcpyHostToDevice, stream2);
    cudaMalloc(&qy_dev2, sizeof(double)*n);
    cudaMemcpyAsync(qy_dev2, qy_h2, sizeof(double)*n, cudaMemcpyHostToDevice, stream2);
    cudaMalloc(&qz_dev2, sizeof(double)*n);
    cudaMemcpyAsync(qz_dev2, qz_h2, sizeof(double)*n, cudaMemcpyHostToDevice, stream2);
    
    cudaMalloc(&vx_dev2, sizeof(double)*n);
    cudaMemcpyAsync(vx_dev2, vx_h2, sizeof(double)*n, cudaMemcpyHostToDevice, stream2);
    cudaMalloc(&vy_dev2, sizeof(double)*n);
    cudaMemcpyAsync(vy_dev2, vy_h2, sizeof(double)*n, cudaMemcpyHostToDevice, stream2);
    cudaMalloc(&vz_dev2, sizeof(double)*n);
    cudaMemcpyAsync(vz_dev2, vz_h2, sizeof(double)*n, cudaMemcpyHostToDevice, stream2);
    
    cudaMalloc(&m_dev2, sizeof(double)*n);
    cudaMemcpyAsync(m_dev2, m_h2, sizeof(double)*n, cudaMemcpyHostToDevice, stream2);
    
    cudaMalloc(&t_dev2, sizeof(int)*n);
    cudaMemcpyAsync(t_dev2, t, sizeof(int)*n, cudaMemcpyHostToDevice, stream2);
    
    
    bool P2 = true;
    
   for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            //cudaSetDevice(0);
            run_step_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(step, qx_dev1, qy_dev1, qz_dev1, vx_dev1, vy_dev1, vz_dev1, m_dev1, t_dev1);
            if(P2)
                run_step_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(step, qx_dev2, qy_dev2, qz_dev2, vx_dev2, vy_dev2, vz_dev2, m_dev2, t_dev2);
            //cudaDeviceSynchronize();
        }
       
       cudaMemcpyAsync(qx_h1, qx_dev1, sizeof(double)*n, cudaMemcpyDeviceToHost, stream1);
       cudaMemcpyAsync(qy_h1, qy_dev1, sizeof(double)*n, cudaMemcpyDeviceToHost, stream1);
       cudaMemcpyAsync(qz_h1, qz_dev1, sizeof(double)*n, cudaMemcpyDeviceToHost, stream1);
       
        double dx = qx_h1[planet] - qx_h1[asteroid];
        double dy = qy_h1[planet] - qy_h1[asteroid];
        double dz = qz_h1[planet] - qz_h1[asteroid];
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
       
       
       if(P2){
           cudaMemcpyAsync(qx_h2, qx_dev2, sizeof(double)*n, cudaMemcpyDeviceToHost, stream2);
           cudaMemcpyAsync(qy_h2, qy_dev2, sizeof(double)*n, cudaMemcpyDeviceToHost, stream2);
           cudaMemcpyAsync(qz_h2, qz_dev2, sizeof(double)*n, cudaMemcpyDeviceToHost, stream2);
           
           double dx = qx_h2[planet] - qx_h2[asteroid];
           double dy = qy_h2[planet] - qy_h2[asteroid];
           double dz = qz_h2[planet] - qz_h2[asteroid];
           if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
               hit_time_step = step;
               P2 = false;
           }

       }
       
       
       
    }
    
    cudaFree(qx_dev1);
    cudaFree(qy_dev1);
    cudaFree(qz_dev1);
    cudaFree(vx_dev1);
    cudaFree(vy_dev1);
    cudaFree(vz_dev1);
    cudaFree(m_dev1);
    cudaFree(t_dev1);
    
    cudaFree(qx_dev2);
    cudaFree(qy_dev2);
    cudaFree(qz_dev2);
    cudaFree(vx_dev2);
    cudaFree(vy_dev2);
    cudaFree(vz_dev2);
    cudaFree(m_dev2);
    cudaFree(t_dev2);
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    

    // Problem 3
    // TODO
    int gravity_device_id = -999;
    double missile_cost = DBL_MAX;

    double *qx_dev3, *qy_dev3, *qz_dev3, *vx_dev3, *vy_dev3, *vz_dev3, *m_dev3;
    int *t_dev3;
        
    cudaSetDevice(1);
        
    
    cudaMalloc(&qx_dev3, sizeof(double)*n);
    cudaMalloc(&qy_dev3, sizeof(double)*n);
    cudaMalloc(&qz_dev3, sizeof(double)*n);
    cudaMalloc(&vx_dev3, sizeof(double)*n);
    cudaMalloc(&vy_dev3, sizeof(double)*n);
    cudaMalloc(&vz_dev3, sizeof(double)*n);
    cudaMalloc(&m_dev3, sizeof(double)*n);
    cudaMalloc(&t_dev3, sizeof(int)*n);
        
    int hitNum = 0;
        
    for(int i=0; i<devList.size(); i++){
            
        read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
            
        double *qx_h3 = &qx[0];
        double *qy_h3 = &qy[0];
        double *qz_h3 = &qz[0];
        double *vx_h3 = &vx[0];
        double *vy_h3 = &vy[0];
        double *vz_h3 = &vz[0];
        double *m_h3 = &m[0];
            
            
        cudaMemcpy(qx_dev3, qx_h3, sizeof(double)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(qy_dev3, qy_h3, sizeof(double)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(qz_dev3, qz_h3, sizeof(double)*n, cudaMemcpyHostToDevice);
            
        cudaMemcpy(vx_dev3, vx_h3, sizeof(double)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(vy_dev3, vy_h3, sizeof(double)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(vz_dev3, vz_h3, sizeof(double)*n, cudaMemcpyHostToDevice);
            
        cudaMemcpy(m_dev3, m_h3, sizeof(double)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(t_dev3, t, sizeof(int)*n, cudaMemcpyHostToDevice);
            
        double qx_m, qy_m, qz_m;  // position of missile
        double travelDist = 0;
            
        qx_m = qx[planet];
        qy_m = qy[planet];
        qz_m = qz[planet];
            
        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                run_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(step, qx_dev3, qy_dev3, qz_dev3, vx_dev3, vy_dev3, vz_dev3, m_dev3, t_dev3);
            }
                
            cudaMemcpy(qx_h3, qx_dev3, sizeof(double)*n, cudaMemcpyDeviceToHost);
            cudaMemcpy(qy_h3, qy_dev3, sizeof(double)*n, cudaMemcpyDeviceToHost);
            cudaMemcpy(qz_h3, qz_dev3, sizeof(double)*n, cudaMemcpyDeviceToHost);
            
            qx_m += param::missile_speed * param::dt * project_x(qx_m, qy_m, qz_m, qx_h3[devList[i]], qy_h3[devList[i]], qz_h3[devList[i]]);
            qy_m += param::missile_speed * param::dt * project_y(qx_m, qy_m, qz_m, qx_h3[devList[i]], qy_h3[devList[i]], qz_h3[devList[i]]);
            qz_m += param::missile_speed * param::dt * project_z(qx_m, qy_m, qz_m, qx_h3[devList[i]], qy_h3[devList[i]], qz_h3[devList[i]]);
                
                
            travelDist += param::missile_speed * param::dt;
                
            if(travelDist >= dist(qx_h3[planet], qy_h3[planet], qz_h3[planet], qx_h3[devList[i]], qy_h3[devList[i]], qz_h3[devList[i]])){
                m_h3[devList[i]] = 0;  // device’s mass becomes zero after it is destroyed
                double c = param::get_missile_cost(step * param::dt);
                if(missile_cost > c){
                    missile_cost = c;
                    gravity_device_id = devList[i];
                }
                cudaMemcpy(m_dev3, m_h3, sizeof(double)*n, cudaMemcpyHostToDevice);
            }
                
            // determine hitting
            double dx = qx_h3[planet] - qx_h3[asteroid];
            double dy = qy_h3[planet] - qy_h3[asteroid];
            double dz = qz_h3[planet] - qz_h3[asteroid];
            if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
                hitNum++;
                //printf("%d\n", hitNum);
                break;
            }
            
                
        }
            
    }
        
    if(hitNum == devList.size() || hit_time_step == -2){
        gravity_device_id = -1;
        missile_cost = 0;
    }
        
    cudaFree(qx_dev3);
    cudaFree(qy_dev3);
    cudaFree(qz_dev3);
    cudaFree(vx_dev3);
    cudaFree(vy_dev3);
    cudaFree(vz_dev3);
    cudaFree(m_dev3);
    cudaFree(t_dev3);
        
    

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}
