#include <GL/glew.h>
#include <GL/freeglut.h>

#include <fstream>
#include <regex>
#include <streambuf>
#include <string>
#include <stdio.h>
#include <algorithm>

#include <cudaDefs.h>
#include <FreeImage.h>
#include <imageManager.h>

using namespace std;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

struct Settings {
    int leaders;
    int followers;
    string heighthMap;
    int heightmapGridX;
    int heightmapGridY;
    float leaderRadius;
    float speedFactor;
    string outputFile;

    string getAtribVal(string &str, const string& atribName) {
        smatch m;
        regex rt(atribName + "\":([\\S\\s]+?(?=,|}))");
        regex_search(str, m, rt);
        string val = m.str().substr(atribName.size()+2);
        if (val.at(0) == '\"') {
            val = val.substr(1, val.size()-2);
        }
        return val;
    }

    Settings(const string& path) {
        fstream inStream(path);
        std::string str((std::istreambuf_iterator<char>(inStream)),std::istreambuf_iterator<char>());
        str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());

        this->leaders = std::stoi(getAtribVal(str, "leaders"));
        this->followers = std::stoi(getAtribVal(str, "followers"));
        this->heightmapGridX = std::stoi(getAtribVal(str, "heightmapGridX"));
        this->heightmapGridY = std::stoi(getAtribVal(str, "heightmapGridY"));
        this->leaderRadius = std::stof(getAtribVal(str, "leaderRadius"));
        this->speedFactor = std::stof(getAtribVal(str, "speedFactor"));
        this->heighthMap = getAtribVal(str, "heightmap");
        this->outputFile = getAtribVal(str, "outputFile");
    }
};

int main(int argc, char* argv[]) {
    initializeCUDA(deviceProp);
    Settings settings("../../settings.json");

    return 0;
}