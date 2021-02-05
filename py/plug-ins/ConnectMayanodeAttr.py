import sys

import maya.api.OpenMaya as om


def maya_useNewAPI():
    pass


class ConnectMayanodeAttr(om.MPxCommand):
    COMMAND_NAME = 'connectMayanodeAttr'
    modifier = None
    undoQueue = []

    sourcePlug = om.MPlug()
    destPlug = om.MPlug()
    forceConnection = False

    def __init__(self):
        super(ConnectMayanodeAttr, self).__init__()

    @classmethod
    def isUndoable(cls):
        return True

    @classmethod
    def hasSyntax(cls):
        return True

    @classmethod
    def syntax(cls):
        _syntax = om.MSyntax()

        # pass a the string of the source plug's path
        _syntax.addArg(om.MSyntax.kString)

        # pass a the string of the destination plug's path
        _syntax.addArg(om.MSyntax.kString)

        # pass a boolean as a kwarg for whether or not to disconnect existing connections
        _syntax.addFlag('f', 'forceConnection', om.MSyntax.kBoolean)

        return _syntax

    def doIt(self, args):
        # gather the values from args
        argParser = om.MArgParser(self.syntax(), args)

        sourcePlugStr = argParser.commandArgumentString(0)
        destPlugStr = argParser.commandArgumentString(1)

        sel = om.MSelectionList()
        sel.add(sourcePlugStr)
        sel.add(destPlugStr)
        self.sourcePlug = sel.getPlug(0)
        self.destPlug = sel.getPlug(1)

        if argParser.isFlagSet('f') or argParser.isFlagSet('forceConnection'):
            self.forceConnection = argParser.flagArgumentBool('f', 0)
        else:
            self.forceConnection = False

        sourceFnAttr = om.MFnAttribute(self.sourcePlug.attribute())
        destFnAttr = om.MFnAttribute(self.destPlug.attribute())

        if not sourceFnAttr.connectable:
            self.displayError('{} is not connectable'.format(self.sourcePlug.name()))
            return
        elif not destFnAttr.connectable:
            self.displayError('{} is not connectable'.format(self.destPlug.name()))
            return
        elif not destFnAttr.writable:
            self.displayError('{} is not writable'.format(self.destPlug.name()))
            return

        if sourceFnAttr.affectsWorldSpace or destFnAttr.affectsWorldSpace:
            self.modifier = om.MDagModifier()
        else:
            self.modifier = om.MDGModifier()

        self.redoIt()

    def redoIt(self):
        existingDestPlugConnection = self.destPlug.source()
        if not existingDestPlugConnection.isNull:
            if self.forceConnection:
                self.modifier.disconnect(existingDestPlugConnection, self.destPlug)
            else:
                self.displayError('Could not make connection as {0} is already connected to {1}.'.format(
                    self.destPlug.name(), existingDestPlugConnection.name()))
                return

        self.modifier.connect(self.sourcePlug, self.destPlug)

        self.modifier.doIt()
        self.undoQueue.append(self.modifier)

    def undoIt(self):
        if self.undoQueue:
            self.undoQueue[-1].undoIt()
            self.undoQueue.pop(-1)


def cmdCreator():
    return ConnectMayanodeAttr()


def initializePlugin(mObject):
    mPlugin = om.MFnPlugin(mObject)
    try:
        mPlugin.registerCommand(ConnectMayanodeAttr.COMMAND_NAME, cmdCreator)
    except:
        sys.stderr.write('Failed to register command {}'.format(ConnectMayanodeAttr.COMMAND_NAME))


def uninitializePlugin(mObject):
    mPlugin = om.MFnPlugin(mObject)
    try:
        mPlugin.deregisterCommand(ConnectMayanodeAttr.COMMAND_NAME)
    except:
        sys.stderr.write('Failed to unregister command {}'.format(ConnectMayanodeAttr.COMMAND_NAME))
