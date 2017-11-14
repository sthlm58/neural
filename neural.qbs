import qbs

Project {

    CppApplication {
        Depends { name: "Qt"; submodules: ["core", "widgets" ] }
        cpp.cxxLanguageVersion: "c++17"

        cpp.defines: [ "DOCTEST_CONFIG_DISABLE" ]
        cpp.debugInformation: true

        consoleApplication: true

        Group {
            name: "headers"
            files: "*.h"
        }

        Group {
            name: "sources"
            files: "*.cpp"
        }
    }
}
