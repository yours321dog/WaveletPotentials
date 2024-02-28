#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include "TestLocalProjection.h"

int TestLocalProjection()
{
    TestWHHDForward1D();
    TestLocalProjectionQ3D_div_total('s');
    TestLocalProjectionQ2D_div_total('s');
    return 0;
}

int main(int argc, char* argv[])
{
    printf("argc: %d\n", argc);
    if (argc == 3)
    {
        printf("argv: %s, %s\n", argv[1], argv[2]);
        if (argv[1][0] == 'p')
        {
            //TestSmoke2D_phinosolid_potential(stoi(std::string(argv[2])));
        }
        else if (argv[1][0] == 'd')
        {
            //TestSmoke2D_phinosolid_potential_downsample(stoi(std::string(argv[2])));
        }
    }
    else
    {
        return TestLocalProjection();
    }

    return 0;
}
