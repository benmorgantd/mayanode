import sys

import maya.api.OpenMaya as om


def maya_useNewAPI():
    pass


class SetMayanodeAttr(om.MPxCommand):
    COMMAND_NAME = 'setMayanodeAttr'
    modifier = None
    plug = om.MPlug()
    value = None
    plugType = None
    undoQueue = []

    def __init__(self):
        super(SetMayanodeAttr, self).__init__()

    @classmethod
    def isUndoable(cls):
        return True

    @classmethod
    def hasSyntax(cls):
        return True

    @classmethod
    def syntax(cls):
        _syntax = om.MSyntax()
        # pass a the string of the self.plug's path
        _syntax.addArg(om.MSyntax.kString)
        # pass the self.value to the appropriate flag
        _syntax.addFlag('d', 'double', om.MSyntax.kDouble)
        _syntax.addFlag('b', 'boolean', om.MSyntax.kBoolean)
        _syntax.addFlag('s', 'string', om.MSyntax.kString)
        # floats and integers will be treated internally as doubles
        _syntax.addFlag('f', 'float', om.MSyntax.kDouble)
        _syntax.addFlag('i', 'integer', om.MSyntax.kLong)
        _syntax.addFlag('a', 'angle', om.MSyntax.kAngle)

        return _syntax

    def doIt(self, args):
        # gather the values from args
        argParser = om.MArgParser(self.syntax(), args)

        plugString = argParser.commandArgumentString(0)
        sel = om.MSelectionList()
        sel.add(plugString)
        self.plug = sel.getPlug(0)

        attributeMObj = self.plug.attribute()
        fnAttr = om.MFnAttribute(attributeMObj)
        if fnAttr.affectsWorldSpace:
            self.modifier = om.MDagModifier()
        else:
            self.modifier = om.MDGModifier()

        if not self.plug.source().isNull:
            self.displayError('{} is connected and cannot be set.'.format(self.plug.name()))
            return

        if argParser.isFlagSet('double') or argParser.isFlagSet('d'):
            self.plugType = 'double'
            self.value = argParser.flagArgumentDouble('d', 0)
            self.redoIt()

        elif argParser.isFlagSet('boolean') or argParser.isFlagSet('b'):
            self.plugType = 'boolean'
            self.value = argParser.flagArgumentBool('b', 0)
            self.redoIt()

        elif argParser.isFlagSet('string') or argParser.isFlagSet('s'):
            self.plugType = 'string'
            self.value = argParser.flagArgumentString('s', 0)
            self.redoIt()

        elif argParser.isFlagSet('float') or argParser.isFlagSet('f'):
            self.plugType = 'float'
            self.value = argParser.flagArgumentFloat('f', 0)
            self.redoIt()

        elif argParser.isFlagSet('int') or argParser.isFlagSet('i'):
            self.plugType = 'int'
            self.value = argParser.flagArgumentInt('i', 0)
            self.redoIt()

        elif argParser.isFlagSet('angle') or argParser.isFlagSet('a'):
            self.plugType = 'angle'
            self.value = argParser.flagArgumentMAngle('a', 0)
            self.redoIt()

    def redoIt(self):
        if self.plugType == 'double':
            self.modifier.newPlugValueDouble(self.plug, self.value)
        elif self.plugType == 'boolean':
            self.modifier.newPlugValueBool(self.plug, self.value)
        elif self.plugType == 'string':
            self.modifier.newPlugValueString(self.plug, self.value)
        elif self.plugType == 'float':
            self.modifier.newPlugValueFloat(self.plug, self.value)
        elif self.plugType == 'int':
            self.modifier.newPlugValueInt(self.plug, self.value)

        self.modifier.doIt()

        self.undoQueue.append(self.modifier)

    def undoIt(self):
        if self.undoQueue:
            self.undoQueue[-1].undoIt()
            self.undoQueue.pop(-1)


def cmdCreator():
    return SetMayanodeAttr()


def initializePlugin(mObject):
    mPlugin = om.MFnPlugin(mObject)
    try:
        mPlugin.registerCommand(SetMayanodeAttr.COMMAND_NAME, cmdCreator)
    except:
        sys.stderr.write('Failed to register command {}'.format(SetMayanodeAttr.COMMAND_NAME))


def uninitializePlugin(mObject):
    mPlugin = om.MFnPlugin(mObject)
    try:
        mPlugin.deregisterCommand(SetMayanodeAttr.COMMAND_NAME)
    except:
        sys.stderr.write('Failed to unregister command {}'.format(SetMayanodeAttr.COMMAND_NAME))
