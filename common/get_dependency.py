# /usr/bin/env python
import pip
import os, sys
import pkg_resources
import inspect, importlib as implib

class BuildRequirments:
    """read existing requirement settings in text file, update version change
        add new package import in script_ and write version info into requirement file
        #default requirment for pip : pip_version == '9.0.1'
        e.g. generator = BuildRequirments(filepath)
             generator.build_requirements()
    """
    def __init__(self, file_path):
        self.dir_filepath = os.path.dirname(file_path)
        self.file_name = os.path.basename(file_path)
        self.file_requirment = os.path.join(self.dir_filepath, "requirements.txt")

    def build_requirements(self):
        requirements = self.load_requirement()
        imported_modules = self.get_import_package_names()
        module_versions = list(self.get_package_dependants(imported_modules))
        for mod_info in module_versions:
            for module in mod_info:
                version = mod_info[module]
                if module in requirements:
                    req_version = requirements[module]
                    if version != module:
                        requirements[module] = version
                else:
                    requirements[module] = version
        self.write_requirment(requirements) 

    def load_requirement(self):
        requirements = {}
        try:
            with open(self.file_requirment, "rb") as infile:
                for line in infile:
                    mod, version = line.split("==")
                    requirements[mod] = version.strip()
        except:
            open(self.file_requirment, "w+").close()
        return requirements

    def write_requirment(self, requirements):
        with open(self.file_requirment, "w") as outfile:
            for mod in requirements:
                version = requirements[mod]
                mod_version = mod + "==" + str(version)
                outfile.write(mod_version + os.linesep)
        outfile.close()

    def get_import_package_names(self):
        pkg_in_use = []
        module_name = (self.dir_filepath + "/" + self.file_name.split(".")[0]).replace("/",".")
        modules = implib.import_module(module_name)
        for i in inspect.getmembers(modules, inspect.ismodule):
            lib_name = str(i[1]).split(" ")[1]
            lib_name = lib_name.split(".")[0]
            pkg_in_use.append(lib_name.replace("'", ''))
        return pkg_in_use

    def get_package_dependants(self, package_):
        for pkg in pip.get_installed_distributions() :
            module = pkg.key
            if module in (set(package_) | set(globals())):
                yield {module: pkg.version}
            if module in sys.modules:
                try:
                    yield {module: sys.modules[module].__version__}
                except:
                    try:
                        if type(modules[module].version) is str:
                            yield {module :sys.modules[module].version}
                        else:
                            yield {module: sys.modules[module].version()}
                    except:
                        try:
                            yield {module: sys.modules[module].VERSION}
                        except:
                            pass


