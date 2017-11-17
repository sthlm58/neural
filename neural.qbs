import qbs

Project {

    CppApplication {
        cpp.cxxLanguageVersion: "c++17"

        cpp.defines: [ "DOCTEST_CONFIG_DISABLE" ]
        cpp.debugInformation: true

        consoleApplication: true

        cpp.includePaths: [ "3rdparty" ]

        Group {
            name: "3rdparty"
            prefix: "3rdparty/"
            files: "*.h"
        }
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
