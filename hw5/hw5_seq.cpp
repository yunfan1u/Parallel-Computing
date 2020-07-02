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
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

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

void run_step(int step, int n, std::vector<double>& qx, std::vector<double>& qy,
    std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
    std::vector<double>& vz, const std::vector<double>& m,
    const std::vector<std::string>& type) {
    // compute accelerations
    std::vector<double> ax(n), ay(n), az(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            if (type[j] == "device") {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax[i] += param::G * mj * dx / dist3;
            ay[i] += param::G * mj * dy / dist3;
            az[i] += param::G * mj * dz / dist3;
        }
    }

    // update velocities
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * param::dt;
        vy[i] += ay[i] * param::dt;
        vz[i] += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
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
    int n, planet, asteroid;  // n: n bodies
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;

    auto distance = [&](int i, int j) -> double {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };

    // Problem 1
    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            m[i] = 0;    // No device
        }
    }
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }

    // Problem 2
    int hit_time_step = -2;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
            hit_time_step = step;
            break;
        }
    }

    // Problem 3
    // TODO
    int gravity_device_id = -999;
    double missile_cost = DBL_MAX;
    
    vector<int> devList;  // device list
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            devList.push_back(i);
        }
    }
    
    int hitNum = 0;
    
    for(int i=0; i<devList.size(); i++){
        
        read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
        
        double qx_m, qy_m, qz_m;  // position of missile
        double travelDist = 0;
        
        qx_m = qx[planet];
        qy_m = qy[planet];
        qz_m = qz[planet];
        
        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
            }
        
            qx_m += param::missile_speed * param::dt * project_x(qx_m, qy_m, qz_m, qx[devList[i]], qy[devList[i]], qz[devList[i]]);
            qy_m += param::missile_speed * param::dt * project_y(qx_m, qy_m, qz_m, qx[devList[i]], qy[devList[i]], qz[devList[i]]);
            qz_m += param::missile_speed * param::dt * project_z(qx_m, qy_m, qz_m, qx[devList[i]], qy[devList[i]], qz[devList[i]]);
            
            
            travelDist += param::missile_speed * param::dt;
            
            if(travelDist >= dist(qx[planet], qy[planet], qz[planet], qx[devList[i]], qy[devList[i]], qz[devList[i]])){
                m[devList[i]] = 0;  // device’s mass becomes zero after it is destroyed
                double c = param::get_missile_cost(step * param::dt);
                if(missile_cost > c){
                    missile_cost = c;
                    gravity_device_id = devList[i];
                }
            }
            
            // determine hitting
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
                hitNum++;
                printf("%d\n", hitNum);
                break;
            }
        
            
        }  // step done
        
        
    }
    
    if(hitNum == devList.size() || hit_time_step == -2){
        gravity_device_id = -1;
        missile_cost = 0;
    }
    
    // done
    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}
