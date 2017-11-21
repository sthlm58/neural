import qbs

Project {

    property bool withTests: false

    CppApplication {
        Depends { name: "Qt"; submodules: ["core", "widgets" ] }

        cpp.defines: [ !withTests ? "DOCTEST_CONFIG_DISABLE" : "" ]
        cpp.includePaths: [ "3rdparty" ]
        cpp.cxxLanguageVersion: "c++17"
        cpp.debugInformation: true

        consoleApplication: true

        Group {
            name: "3rdparty"
            prefix: "3rdparty/"
            files: "*.h"
        }

        Group {
            name: "headers"
            files: [
                "*.h",
            ]
        }

        Group {
            name: "sources"
            files: [
                "*.cpp",
            ]
            excludeFiles: parent.withTests ? "main.cpp" : "main_test.cpp"
        }
    }


}
