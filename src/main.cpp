#include <CloverBase.h>
#include "../lib/cxxopts.h"
#include "../lib/sysinfo.h"

#include "../test/search/00_search.h"
#include "../test/performance/00_test.h"
#include "../test/validate/00_validate.h"
#include "../test/accuracy/00_accuracy.h"

int main (int argc, const char* argv[])
{
    //
    // Parse the option and perform different tests
    //
    try {
        cxxopts::Options options(argv[0], " - example command line options");
        options.positional_help("[optional args]").show_positional_help();
        options.add_options()
                ("a,accuracy", "Start the tests for accuracy")
                ("g,grid", "Start the grid search for hyperparameter optimization")
                ("v,validate", "Start the tests for validation")
                ("p,performance", "Start the tests for performance")
                ("h,help", "Print help")
        ;
        options.parse_positional({"input", "output", "positional"});
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }

        print_compiler_and_system_info();
        CloverBase::initializeLibraries();

        if (result.count("a")) {
            test_accuracy();
        }

        if (result.count("g")) {
            search(argc, argv);
        }

        if (result.count("v")) {
            validate(argc, argv);
        }

        if (result.count("p")) {
            test(argc, argv);
        }

    } catch (const cxxopts::OptionException& e)  {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
    return 0;
}