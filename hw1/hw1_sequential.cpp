#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <regex>
#include <tuple>
#include <boost/functional/hash.hpp>
//#include "tbb/concurrent_unordered_set.h"
#include <unordered_set>
#include <boost/unordered_set.hpp>
#include <array>

using namespace std;

typedef vector<vector<char>> Table;
 
vector<vector<char>> sData, dData;  // s: solution  ,d
int x, y;
vector<int> arrX, arrY;

void input(string file)
{
    ifstream ifile(file.c_str());
    string line;
    int l = 0;

    while(getline(ifile, line)){
        vector<char> sTemp, dTemp;
        for(int i=0; i<line.size(); i++){
            char s = ' ', d = ' ', c = line[i];
            
            if(c == '#'){
                s = '#';
                d = '#';
            }
            else if(c == '.' || c == 'X' || c == 'O'){  // X: box on target, O: player on target
                s = '.';
                arrX.push_back(i);
                arrY.push_back(l);
            }
            if(c == 'o' || c == 'O'){
                d = 'o';
                x = i;
                y = l;
            }
            else if(c == 'x' || c == 'X')
                d = 'X';

            sTemp.push_back(s);
            dTemp.push_back(d);
        }
        sData.push_back(sTemp);
        dData.push_back(dTemp);
        l++;
    }
    
}

class Board{
public:
    int px, py;

    Board(int x, int y){
        px = x;
        py = y;
    }

    bool isDead(int x, int y, const Table &data){

        if((data[y-1][x]=='#' && data[y][x+1]=='#') ||
            (data[y][x+1]=='#' && data[y+1][x]=='#') ||
            (data[y+1][x]=='#' && data[y][x-1]=='#')  ||
            (data[y][x-1]=='#' && data[y-1][x]=='#'))
            return true;
        
        if((data[y][x+1]=='X' && data[y-1][x]=='#' && data[y-1][x+1]=='#') || (data[y][x-1]=='X' && data[y-1][x]=='#' && data[y-1][x-1]=='#') ||
           (data[y][x+1]=='X' && data[y+1][x]=='#' && data[y+1][x+1]=='#') || (data[y][x-1]=='X' && data[y+1][x]=='#' && data[y+1][x-1]=='#') ||
           (data[y-1][x]=='X' && data[y][x-1]=='#' && data[y-1][x-1]=='#') || (data[y+1][x]=='X' && data[y][x-1]=='#' && data[y+1][x-1]=='#') ||
           (data[y-1][x]=='X' && data[y][x+1]=='#' && data[y-1][x+1]=='#') || (data[y+1][x]=='X' && data[y][x+1]=='#' && data[y+1][x+1]=='#'))
            return true;
        
        return false;
    }
    
    bool move(int x, int y, int dx, int dy, Table &data){
        if(data[y+dy][x+dx] != ' ')
            return false;
    
        data[y][x] = ' ';
        data[y+dy][x+dx] = 'o';
        return true;
    }
    
    bool push(int x, int y, int dx, int dy, Table &data){
        if( data[y+2*dy][x+2*dx] != ' ')
            return false;
        if(isDead(x+2*dx, y+2*dy, data) && sData[y+2*dy][x+2*dx] != '.')
            return false;

        data[y][x] = ' ';
        data[y+dy][x+dx] = 'o';
        data[y+2*dy][x+2*dx] = 'X';

        return true;
    }

    bool isSolved(const Table &data){
        for(int i=0; i<arrX.size(); i++){
            if(data[arrY[i]][arrX[i]] != 'X')
                return false;
        }
        return true;
    }

    
    string solve()
    {
        unordered_set<Table, boost::hash<Table>> visited;
        queue<tuple<Table, string, int, int>> open;
    
        open.push(make_tuple(dData, "", px, py));
        visited.insert(dData);
    
        array<tuple<int, int, char>, 4> dirs;
        dirs[0] = make_tuple(0, -1, 'W');
        dirs[1] = make_tuple(1, 0, 'D');
        dirs[2] = make_tuple(0, 1, 'S');
        dirs[3] = make_tuple(-1, 0, 'A');
    
        while(open.size() > 0)
        {
            Table temp;
            Table cur = get<0>(open.front());
            
            string cSol = get<1>(open.front());
            int x = get<2>(open.front());
            int y = get<3>(open.front());
            
            open.pop();
    
            // BFS
            bool cancel = false;
            int _i;

            for(int i=0; i<4; ++i){
                temp = cur;
                int dx = get<0>(dirs[i]);
                int dy = get<1>(dirs[i]);
    
                if(temp[y+dy][x+dx] == 'X'){  // Push
                    if(push(x, y, dx, dy, temp) && (visited.find(temp) == visited.end())){ // not in the set
                        if(isSolved(temp)){
                            return cSol + get<2>(dirs[i]);
                        }
                        open.push(make_tuple(temp, cSol + get<2>(dirs[i]), x+dx, y+dy));
                        visited.insert(temp);
                    }
                }
                else if(move(x, y, dx, dy, temp) && (visited.find(temp) == visited.end())){  // Move
                    open.push(make_tuple(temp, cSol + get<2>(dirs[i]), x+dx, y+dy));
                    visited.insert(temp);
                }
            }

        }
    
        return "No solution";
    }

};

int main(int argc, char *argv[]){

    input(argv[1]);
    Board b(x, y);
 
    cout << b.solve() << endl;
    return 0;
}
