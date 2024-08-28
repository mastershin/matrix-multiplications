#pragma once

#include <cassert>
#include <chrono>
#include <iostream>
#include <tuple>
#include <map>
#include <unordered_map>
#include <string>   // for stoi

using ArgsMap = std::unordered_map<std::string, std::string>;

ArgsMap parse_command_args(int argc, char *argv[])
{
    // using ArgsMap = std::unordered_map<std::string, std::string>
    ArgsMap args_map;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0)
        { // Check if arg starts with '--'
            std::string key = arg.substr(2);
            args_map[key] = "";
            if (i + 1 < argc && argv[i + 1][0] != '-')
            { // Next arg is the value
                args_map[key] = argv[++i];
            }
        }
    }
    return args_map;
}

void die(const std::string &msg)
{
  std::cerr << msg << std::endl;
  exit(1);
}