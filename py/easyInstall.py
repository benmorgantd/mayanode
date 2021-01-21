import os

import maya.cmds as cmds

# TODO: remove maya dependencies on this install (make an exe or bat file or something)

MODULE_NAME = 'mayanode'

modString = """
+ PLATFORM:win64 {0} 1.0 MODULE_LOC
MYMODULE_SHADERS_LOCATION:=shaders 
MYMODULE_INCLUDE_LOCATION:=shaders/include 
MOD_PLUG_IN_PATH+:=bin/image 
PATH+:=bin 
icons: ../icons 
presets: ../presets 
scripts: ../scripts 

+ PLATFORM:mac {0} 1.0 MODULE_LOC 
MYMODULE_SHADERS_LOCATION:=shaders 
MYMODULE_INCLUDE_LOCATION:=shaders/include 
MOD_PLUG_IN_PATH+:=bin/image 
PATH+:=bin 
icons: ../icons 
presets: ../presets 
scripts: ../scripts 
""".format(MODULE_NAME)


def easyInstall():
    mayaAppDir = cmds.internalVar(userAppDir=1)
    try:
        # have the user browse for the package directory on their system
        packageDir = cmds.fileDialog2(okCaption="Set Path", fileMode=3, caption="Browse to the Scripts Folder")[0]
    except TypeError:
        # Will happen if the user exits the file dialog
        failMessage = "Install of the {0} module failed.\n\nPlease drag the file back into Maya to install.".format(MODULE_NAME)
        print("\n" + failMessage)
        cmds.confirmDialog(title="Error", message=failMessage)
        return

    print('Package directory set to: {0}'.format(packageDir))

    # make the \modules directory if it doesn't already exist
    if not os.path.exists(os.path.join(mayaAppDir, "modules")):
        os.makedirs(os.path.join(mayaAppDir, "modules"))

    # create our mod file in the user's maya modules directory
    modFile = open(os.path.join(mayaAppDir, "modules", "{0}.mod".format(MODULE_NAME)), "w+")

    writeString = modString.replace("MODULE_LOC", packageDir)

    # write the contents of our package file
    modFile.write(writeString)
    modFile.close()

    successMessage = "Successfully installed the {0} module.\n\nPlease restart Maya to finish installation.".format(MODULE_NAME)
    print("\n" + successMessage)
    cmds.confirmDialog(title="Success", message=successMessage)


easyInstall()
