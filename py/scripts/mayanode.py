import sys
import inspect
import json
import os

import maya.cmds as cmds
import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as omAnim

__all__ = [className for className, obj in inspect.getmembers(sys.modules[__name__]) if inspect.isclass(obj)]

def maya_useNewAPI():
    pass

# CONSTANTS #

ROTATE_ORDERS = [om.MEulerRotation.kXYZ, om.MEulerRotation.kYZX, om.MEulerRotation.kZXY, om.MEulerRotation.kXZY,
                 om.MEulerRotation.kYXZ, om.MEulerRotation.kZYX]

DISPLAY_COLORS = {'black': 1, 'grey': 3, 'blue': 6, 'pink': 9, 'red': 13, 'lime': 14, 'darkblue': 15, 'white': 16,
                  'yellow': 17, 'green': 23}

SPACES = {'world': om.MSpace.kWorld, 'object': om.MSpace.kObject, 'local': om.MSpace.kTransform}

ORIGIN_POINT = om.MPoint.kOrigin
ORIGIN_POINT_FLOAT = om.MFloatPoint.kOrigin

NULL_OBJECT = om.MObject.kNullObj
DG_MODIFIER = om.MDGModifier()
DAG_MODIFIER = om.MDagModifier()

DAG_NODE_TYPES = ['transform', 'mesh', 'locator']
INVALID_NODE_TYPES = ['shape', 'curve']

PICK_MATRIX_PARTS = {'useTranslate', 'useRotate', 'useScale', 'useShear'}
CONSTRAINT_TYPE_USAGES = {'parent': {}, 'point': {'useRotate': False, 'useScale': False, 'useShear': False},
                          'orient': {'useTranslate': False, 'useScale': False, 'useShear': False},
                          'scale': {'useTranslate': False, 'useRotate': False},
                          'aim': {'useTranslate': False, 'useScale': False, 'useShear': False}}

INFINITY = sys.float_info.max


class Scene(object):
    @classmethod
    def stringToMObject(cls, sourceString):
        _mObject = None
        if isinstance(sourceString, basestring):
            try:
                # we only want to deal in objects that have unique names
                _sel = om.MSelectionList()
                _sel.add(sourceString)
                _mObject = _sel.getDependNode(0)
            except RuntimeError:
                raise RuntimeError('Object {0} does not exist or is not unique.'.format(sourceString))
        else:
            return None

        return _mObject

    @classmethod
    def getBoundingBox(cls):
        """Returns the bounding box around the DAGs in the scene"""

        boundingBox = om.MBoundingBox()

        itDag = om.MItDag(om.MItDag.kDepthFirst, om.MFn.kMesh)

        while not itDag.isDone():
            dag = om.MDagPath()
            itDag.getPath(dag)

            fnDag = om.MFnDagNode(dag)
            dagBoundingBox = fnDag.boundingBox()
            dagBoundingBox.transformUsing(dag.inclusiveMatrix())

            boundingBox.expand(dagBoundingBox)

            itDag.next()

        return boundingBox

    @classmethod
    def getSelectedVertexIDs(cls):
        '''If a face or edge, converts to a vertex.'''

        # TODO: support conversion to CVs

        sel = om.MGlobal.getActiveSelectionList()
        dag, selectedComponents = sel.getComponent(0)

        vertexConversion = set()

        if selectedComponents.apiType() == om.MFn.kMeshVertComponent:
            return dag, selectedComponents
        elif selectedComponents.apiType() == om.MFn.kMeshEdgeComponent:
            edgeIter = om.MItMeshEdge(dag, selectedComponents)
            while not edgeIter.isDone():
                vert1 = edgeIter.vertexId(0)
                vert2 = edgeIter.vertexId(1)
                vertexConversion.add(vert1)
                vertexConversion.add(vert2)
                edgeIter.next()
        elif selectedComponents.apiType() == om.MFn.kMeshPolygonComponent:
            faceIter = om.MItMeshPolygon(dag, selectedComponents)
            while not faceIter.isDone():
                connectedVertices = faceIter.getVertices()
                vertexConversion.update(connectedVertices)
                faceIter.next()
        elif selectedComponents.apiType() == om.MFn.kInvalid:
            # no selected components. A transform might only be selected
            _dag = om.MDagPath(dag)
            _dag.extendToShape(0)
            # check what the shape's type is
            _shapeMObj = _dag.node()
            if _shapeMObj.hasFn(om.MFn.kMesh):
                _mesh = Mesh(_shapeMObj)
                if not _mesh.fnDagNode.isIntermediateObject:
                    vertexConversion.update(xrange(_mesh.fnMesh.numVertices))
            else:
                return

        newSel = om.MSelectionList()
        for vertId in vertexConversion:
            newSel.add(dag.fullPathName() + '.vtx[{0}]'.format(vertId))

        dag, selectedComponents = newSel.getComponent(0)

        return dag, selectedComponents

    @classmethod
    def getTypesFromSelection(cls):
        sel = om.MGlobal.getActiveSelectionList()

        selectionTypes = []

        itSel = om.MItSelectionList(sel)
        while not itSel.isDone():
            selectionTypes.append(itSel.getDependNode().apiType())
            itSel.next()

        return selectionTypes

    @classmethod
    def createNode(cls, nodeType, nodeName):
        _modifier = DG_MODIFIER

        try:
            _nodeMObject = _modifier.createNode(nodeType)
        except TypeError:
            # Could be a DAG node type, or an unknown node type
            try:
                _modifier = DAG_MODIFIER
                _nodeMObject = _modifier.createNode(nodeType)
            except:
                raise

        _modifier.renameNode(_nodeMObject, nodeName)
        _modifier.doIt()

        return _nodeMObject

    @classmethod
    def setAttr(cls, attribute, value):
        raise NotImplementedError()

    @classmethod
    def mayaVersion(cls):
        return om.MGlobal.mayaVersion()


class DependencyNode(object):
    @classmethod
    def stringToMObject(cls, sourceString):
        _mObject = None
        if isinstance(sourceString, basestring):
            try:
                # we only want to deal in objects that have unique names
                _sel = om.MSelectionList()
                _sel.add(sourceString)
                _mObject = _sel.getDependNode(0)
            except RuntimeError:
                raise RuntimeError('Object {0} does not exist or is not unique.'.format(sourceString))
        else:
            return None

        return _mObject

    @classmethod
    def connect(cls, sourcePlug, destPlug, force=True):
        assert isinstance(sourcePlug, om.MPlug)
        assert isinstance(destPlug, om.MPlug)

        if sourcePlug.isNull or destPlug.isNull:
            raise ValueError('Cannot make connection as one of the given plugs is null.')

        cmds.connectMayanodeAttr(sourcePlug.name(), destPlug.name(), forceConnection=force)

    @classmethod
    def disconnect(cls, sourcePlug, destPlug):
        assert isinstance(sourcePlug, om.MPlug)
        assert isinstance(destPlug, om.MPlug)

        if sourcePlug.isNull or destPlug.isNull:
            raise ValueError('Cannot make disconnection as one of the given plugs is null.')

        DG_MODIFIER.disconnect(sourcePlug, destPlug)
        DG_MODIFIER.doIt()

    def __init__(self, seed):
        self._initialized = False
        # I think many OpenMaya objects inherit from MObject, so we may want to directly check types instead
        if isinstance(seed, om.MObject):
            if seed.isNull():
                raise ValueError('Cannot initialize {} as the given MObject is null.'.format(seed))
            self.mObject = om.MObject(seed)
        elif isinstance(seed, basestring):
            self.mObject = self.stringToMObject(seed)
        else:
            raise RuntimeError(
                'Cannot initialize a mayanode with the given seed {0} of type {1}'.format(seed, type(seed)))

        self.fnDependencyNode = om.MFnDependencyNode(self.mObject)

        self.checkType()

        self._initialized = True

    def __repr__(self):
        # return '<{0} {1} object {2}>'.format(__name__, type(self).__name__, self.fnDependencyNode.name())
        return self.fnDependencyNode.name()

    def __str__(self):
        return self.fnDependencyNode.name()

    def __add__(self, other):
        return self.fnDependencyNode.name() + other

    def __radd__(self, other):
        return other + self.fnDependencyNode.name()

    def __eq__(self, other):
        if isinstance(other, DependencyNode):
            return self.mObject == other.mObject
        else:
            return False

    @property
    def mFnType(self):
        return om.MFn.kDependencyNode, 'kDependencyNode'

    @property
    def name(self):
        return self.fnDependencyNode.name()

    @name.setter
    def name(self, value):
        self.fnDependencyNode.setName(value)

    def checkType(self):
        _mFnType = self.mFnType
        if not self.mObject.hasFn(_mFnType[0]):
            raise TypeError('Give object {0} of type {1} is not a subclass of {2}.'.format(self.fnDependencyNode.name(),
                                                                                           self.mObject.apiTypeStr,
                                                                                           _mFnType[1]))
        elif self.mObject.hasFn(om.MFn.kDagNode) and type(self) == DependencyNode:
            raise TypeError('Cannot initialize {} as a DependencyNode as it is a DagNode.'.format(
                self.fnDependencyNode.name()))
        elif self.mObject.hasFn(om.MFn.kTransform) and type(self) == DagNode:
            raise TypeError('Cannot initialize {} as a DagNode as it is a Transform.'.format(
                self.fnDependencyNode.name()))

    @property
    def metadata(self):
        if self.fnDependencyNode.hasAttribute('metadata'):
            return json.loads(self.metadata)
        else:
            return {}

    @metadata.setter
    def metadata(self, value):
        # use json.dumps. Make this value a json dictionary
        if 'metadata' not in cmds.listAttr(self.name):
            cmds.addAttr(self.name, longName='metadata', dataType='string')

        cmds.setAttr(self.name + '.metadata', json.dumps(value), type='string')

    def get(self, attribute):
        if hasattr(self, 'fnDependencyNode') and self.fnDependencyNode.hasAttribute(attribute):
            plug = self.findPlug(attribute)
            attributeMObj = plug.attribute()
            plugType = attributeMObj.apiType()

            if plugType == om.MFn.kDoubleLinearAttribute:
                return plug.asDouble()
            elif plugType == om.MFn.kDoubleAngleAttribute:
                return plug.asMAngle()
            elif plugType == om.MFn.kNumericAttribute:
                # get data type
                fnNumericAttr = om.MFnNumericAttribute(attributeMObj)
                unitType = fnNumericAttr.numericType()
                if unitType == 1:
                    return plug.asBool()
                elif unitType == 3:
                    return plug.asChar()
                elif unitType == 4:
                    return plug.asShort()
                elif unitType == 7:
                    return plug.asLong()
                elif unitType == 8:
                    return plug.asInt()
                elif unitType == 11:
                    return plug.asFloat()
                elif unitType in [13, 14]:
                    return plug.asFloat()
                else:
                    raise AttributeError('The given MFnNumericData type {0} is not supported.'.format(unitType))
            elif plugType == om.MFn.kAttribute3Double:
                _result = []
                for i in xrange(plug.numChildren()):
                    _childPlug = plug.child(i)
                    _result.append(_childPlug.asDouble())
                return _result
            elif plugType == om.MFn.kAttribute3Float:
                _result = []
                for i in xrange(plug.numChildren()):
                    _childPlug = plug.child(i)
                    _result.append(_childPlug.asFloat())
                return _result
            elif plugType == om.MFn.kTypedAttribute:
                fnTypedAttr = om.MFnTypedAttribute(attributeMObj)
                typedAttrType = fnTypedAttr.attrType()
                if typedAttrType == 4:
                    return plug.asString()
                elif typedAttrType == 5:
                    # kMatrix type. Get matrix data without the MFnTransform
                    if plug.isArray:
                        # worldMatrix, worldInverseMatrix, parentMatrix, parentInverseMatrix
                        _childPlug = plug.elementByLogicalIndex(0)
                        fnMatrixData = om.MFnMatrixData(_childPlug.asMObject())
                        return fnMatrixData.transformation()
                    else:
                        # matrix, inverseMatrix, xformMatrix
                        _fnMatrixData = om.MFnMatrixData(plug.asMObject())
                        return _fnMatrixData.transformation()
                else:
                    print(typedAttrType)

                raise AttributeError(
                    'getattr is currently not supported for attribute type {}.'.format(attributeMObj.apiTypeStr))
            elif plugType == om.MFn.kEnumAttribute:
                return plug.asInt()
            elif plugType == om.MFn.kStringData:
                return plug.asString()
            elif plugType == om.MFn.kMessageAttribute:
                raise AttributeError(
                    'getattr is currently not supported for attribute type {}.'.format(attributeMObj.apiTypeStr))
            else:
                raise AttributeError(
                    'getattr is currently not supported for attribute type {}.'.format(attributeMObj.apiTypeStr))

    def set(self, attribute, value):
        if hasattr(self, 'fnDependencyNode') and self.fnDependencyNode.hasAttribute(attribute):
            plug = self.findPlug(attribute)
            attributeMObj = plug.attribute()
            plugType = attributeMObj.apiType()
            fnAttr = om.MFnAttribute(attributeMObj)

            if not plugType == om.MFn.kTypedAttribute and not fnAttr.writable:
                raise AttributeError('Given attribute {0} is not writable.'.format(fnAttr.name()))
            if plugType == om.MFn.kDoubleLinearAttribute:
                cmds.setMayanodeAttr(plug.name(), double=value)
            elif plugType == om.MFn.kDoubleAngleAttribute:
                cmds.setMayanodeAttr(plug, angle=value)
            elif plugType == om.MFn.kNumericAttribute:
                # get data type
                fnNumericAttr = om.MFnNumericAttribute(attributeMObj)
                unitType = fnNumericAttr.numericType()
                if unitType == 1:
                    cmds.setMayanodeAttr(plug.name(), boolean=value)
                elif unitType == 3:
                    # TODO: do we really need a char arg type?
                    cmds.setMayanodeAttr(plug, string=value)
                elif unitType == 4:
                    # TODO: do we really need a short arg type?
                    cmds.setMayanodeAttr(plug, integer=value)
                elif unitType == 7:
                    cmds.setMayanodeAttr(plug, integer=value)
                elif unitType == 8:
                    cmds.setMayanodeAttr(plug, integer=value)
                elif unitType in [13, 14]:
                    cmds.setMayanodeAttr(plug, float=value)
                else:
                    raise AttributeError('The given MFnNumericData type {0} is not supported.'.format(unitType))
            elif plugType == om.MFn.kAttribute3Double:
                if isinstance(value, om.MQuaternion):
                    _eulerRot = value.asEulerRotation()
                    _eulerRot = _eulerRot.reorder(ROTATE_ORDERS[self.get('rotateOrder')])
                    value = _eulerRot
                for i in xrange(plug.numChildren()):
                    _childPlug = plug.child(i)
                    cmds.setMayanodeAttr(_childPlug, float=value[i])

            elif plugType == om.MFn.kTypedAttribute:
                fnTypedAttr = om.MFnTypedAttribute(attributeMObj)
                typedAttrType = fnTypedAttr.attrType()
                if typedAttrType == 5:
                    # Matrix attributes are not writable! You cannot write to them, only read from them.
                    # They cannot be directly connected, which means their values cannot be set
                    # outside of a node's compute function.
                    # Instead, read the data from the matrix and set attributes instead...
                    # NOTE: you cannot set this attribute on anything but self.matrix

                    if fnTypedAttr.array:
                        raise RuntimeError(
                            '{0}.DependencyNode : set() is unsupported for Matrix Array attribute {1}'.format(__name__,
                                                                                                              plug.name()))

                    _xformMatrix = om.MTransformationMatrix(value)
                    _translateVector = _xformMatrix.translation(om.MSpace.kWorld)
                    _rotateQuaternion = _xformMatrix.rotation()
                    _scale = _xformMatrix.scale(SPACES['local'])
                    _shear = _xformMatrix.shear(SPACES['local'])
                    self.set('translate', _translateVector)
                    self.set('rotate', _rotateQuaternion)
                    self.set('scale', _scale)
                    self.set('shear', _shear)

            elif plugType == om.MFn.kEnumAttribute:
                cmds.setMayanodeAttr(plug.name(), integer=value)
            elif plugType == om.MFn.kStringData:
                cmds.setMayanodeAttr(plug.name(), string=value)
            elif plugType == om.MFn.kMessageAttribute:
                raise AttributeError('set() is currently not supported for this attribute type.')
            else:
                raise AttributeError('set() is currently not supported for this attribute type.')

    def findPlug(self, plugString, multiIndex=0):
        if self.fnDependencyNode.hasAttribute(plugString):
            plug = self.fnDependencyNode.findPlug(plugString, False)
            if plug.isArray:
                _childPlug = plug.elementByLogicalIndex(multiIndex)
                return _childPlug
            else:
                return plug
        else:
            raise AttributeError('Object {0} has no plug {1}'.format(self, plugString))

    def getInputConnection(self, plugString):
        plug = self.findPlug(plugString)

        if plug.isDestination:
            return DependencyNode(plug.source().node())
        else:
            return None

    def getFileTexturePath(self, attributeType):
        textureFileNode = self.getInputConnection(attributeType)
        if textureFileNode:
            filePath = os.path.abspath(textureFileNode.get('fileTextureName'))
            if filePath.endswith('Maya{}'.format(Scene.mayaVersion())):
                return None
            else:
                return filePath
        else:
            return None

    def isNodeUpstream(self, sourcePlug, nodeMObject):
        itDG = om.MItDependencyGraph(self.mObject)
        itDG.resetTo(sourcePlug, nodeMObject.apiType(), om.MItDependencyGraph.kUpstream,
                     om.MItDependencyGraph.kDepthFirst, om.MItDependencyGraph.kNodeLevel)

        if itDG.isDone():
            # No nodes upstream of that plug of the given type
            return False
        while not itDG.isDone():
            if itDG.currentNode() == nodeMObject:
                return True

            itDG.next()

    def findUpstreamNodesOfType(self, nodeType, sourcePlug=None):
        itDG = om.MItDependencyGraph(self.mObject)
        if sourcePlug:
            itDG.resetTo(sourcePlug, nodeType, om.MItDependencyGraph.kUpstream, om.MItDependencyGraph.kDepthFirst,
                         om.MItDependencyGraph.kNodeLevel)
        else:
            itDG.resetTo(self.mObject, nodeType, om.MItDependencyGraph.kUpstream, om.MItDependencyGraph.kDepthFirst,
                         om.MItDependencyGraph.kNodeLevel)

        if itDG.isDone():
            # No nodes upstream of that plug of the given type
            return None

        nodeMObjects = []

        while not itDG.isDone():
            nodeMObjects.append(itDG.currentNode())
            itDG.next()

        nodes = []

        # TODO: Create a fn that will create the right mayanode type from a given mObject
        for mObject in nodeMObjects:
            if mObject.hasFn(om.MFn.kTransform):
                nodes.append(Transform(mObject))
            else:
                nodes.append(DependencyNode(mObject))
        return nodes

    def lockAttr(self, attr):
        if self.fnDependencyNode.hasAttribute(attr):
            plug = self.findPlug(attr)
            plug.setLocked(True)

    def unlockAttr(self, attr):
        if self.fnDependencyNode.hasAttribute(attr):
            plug = self.findPlug(attr)
            plug.setLocked(False)

    def hideAttr(self, attr):
        if self.fnDependencyNode.hasAttribute(attr):
            plug = self.findPlug(attr)
            plug.setKeyable(False)

    def showAttr(self, attr):
        if self.fnDependencyNode.hasAttribute(attr):
            plug = self.findPlug(attr)
            plug.setKeyable(True)


class DagNode(DependencyNode):
    def __init__(self, seed):
        super(DagNode, self).__init__(seed)

        self._initialized = False

        _sel = om.MSelectionList()
        try:
            _sel.add(self.fnDependencyNode.name())
            _dag = _sel.getDagPath(0)
            self.dagPath = _dag
        except RuntimeError:
            # object's short name is not unique. Iterate through the dag until we find this mObject
            raise RuntimeError('Object short name is not unique')

        self.fnDagNode = om.MFnDagNode(self.dagPath)

        self._initialized = True

    def __repr__(self):
        return self.dagPath.fullPathName()

    def __str__(self):
        return self.dagPath.fullPathName()

    def __add__(self, other):
        return self.dagPath.fullPathName() + other

    def __radd__(self, other):
        return other + self.dagPath.fullPathName()

    @property
    def mFnType(self):
        return om.MFn.kDagNode, 'kDagNode'

    @property
    def name(self):
        return self.dagPath.fullPathName()

    @name.setter
    def name(self, value):
        self.fnDependencyNode.setName(self.makeShortNameUnique(value))

    @property
    def shortName(self):
        return self.dagPath.partialPathName()

    @shortName.setter
    def shortName(self, value):
        self.fnDependencyNode.setName(value)

    @property
    def parent(self):
        # return the Transform of the parent
        if self.dagPath.length() > 1:
            # TODO: put this in Transform? ...
            return DagNode(self.fnDagNode.parent(0))
        else:
            return None

    @parent.setter
    def parent(self, newParent):
        if not newParent:
            newParent = NULL_OBJECT
        elif isinstance(newParent, DependencyNode):
            newParent = newParent.mObject
        elif isinstance(newParent, basestring):
            newParent = Scene.stringToMObject(newParent)

        dagMod = om.MDagModifier()
        dagMod.reparentNode(self.mObject, newParent)
        dagMod.doIt()

    def children(self, filterType=om.MFn.kInvalid):
        '''Return child transforms'''

        # Filters don't seem to work here
        _itDag = om.MItDag()
        _itDag.reset(self.dagPath, om.MItDag.kDepthFirst, filterType)

        if _itDag.isDone():
            return []
        else:
            # don't give ourself as the first item
            _itDag.next()

        _mObjArray = om.MObjectArray()

        while not _itDag.isDone():
            if _itDag.depth() == 1:
                _currentMObject = _itDag.currentItem()
                if _currentMObject.hasFn(om.MFn.kTransform):
                    _mObjArray.append(_currentMObject)
            _itDag.next()

        _result = []
        for i in xrange(len(_mObjArray)):
            # TODO: put this in Transform? ...
            _result.append(DagNode(_mObjArray[i]))
        return _result

    def allChildren(self, filterType=om.MFn.kInvalid):
        '''Return all descendent children transforms'''

        _itDag = om.MItDag()
        _itDag.reset(self.dagPath, om.MItDag.kDepthFirst, filterType)

        if _itDag.isDone():
            return []
        else:
            # don't give ourself as the first item
            _itDag.next()

        _mObjArray = om.MObjectArray()

        while not _itDag.isDone():
            _currentMObject = _itDag.currentItem()
            if _currentMObject.hasFn(om.MFn.kTransform):
                _mObjArray.append(_currentMObject)
            _itDag.next()

        _result = []
        for i in xrange(len(_mObjArray)):
            # TODO: put this in Transform?
            _result.append(DagNode(_mObjArray[i]))
        return _result

    @classmethod
    def makeShortNameUnique(cls, name):
        _itDag = om.MItDag(om.MItDag.kDepthFirst, om.MFn.kTransform)
        _dag = om.MDagPath()
        _numDuplicates = 0
        _possibleName = name

        while not _itDag.isDone():
            if _numDuplicates:
                _possibleName = '{0}{1}'.format(name, _numDuplicates)
            if _itDag.partialPathName() == _possibleName:
                _numDuplicates += 1
            _itDag.next()

        if _numDuplicates:
            return '{0}{1}'.format(name, _numDuplicates)
        else:
            return name

    def hasParent(self, parent):
        if isinstance(parent, DependencyNode):
            parent = parent.mObject
        elif isinstance(parent, basestring):
            parent = self.stringToMObject(parent)

        if parent:
            return self.fnDagNode.hasParent(parent)
        else:
            return False

    def hasChild(self, child):
        if isinstance(child, DependencyNode):
            child = child.mObject
        elif isinstance(child, basestring):
            child = self.stringToMObject(child)

        if child:
            return self.fnDagNode.hasChild(child)
        else:
            return False


class Transform(DagNode):
    @classmethod
    def matrixToList(cls, matrix):
        if isinstance(matrix, om.MTransformationMatrix):
            matrix = matrix.asMatrix()

        matrixList = []

        for i in xrange(16):
            matrixList.append(matrix[i])

        return matrixList

    def __init__(self, seed):
        super(Transform, self).__init__(seed)

    @property
    def mFnType(self):
        return om.MFn.kTransform, 'kTransform'

    @property
    def fnTransform(self):
        return om.MFnTransform(self.dagPath)

    @property
    def worldPosition(self):
        _worldMatrixList = self.worldMatrixList
        return om.MVector(*_worldMatrixList[-4:-1])

    @worldPosition.setter
    def worldPosition(self, vector):
        _vector = om.MVector(vector[0], vector[1], vector[2])
        self.fnTransform.setTranslation(_vector, om.MSpace.kWorld)

    @property
    def worldRotation(self):
        quat = self.fnTransform.rotation(om.MSpace.kWorld, True)
        return quat

    @worldRotation.setter
    def worldRotation(self, rotation):
        self.fnTransform.setRotation(rotation, om.MSpace.kWorld)

    @property
    def matrixVectors(self):
        matrix = self.matrixList
        matrixVectors = om.MVectorArray()

        for vector in matrix:
            matrixVectors.append(om.MVector(*vector))

        return matrixVectors

    @property
    def matrixList(self):
        return self.matrixToList(self.get('matrix'))

    @property
    def worldMatrixList(self):
        return self.matrixToList(self.get('worldMatrix'))

    @property
    def longAxis(self):
        _translate = self.get('translate')
        axis = ['X', 'Y', 'Z']
        return axis[_translate.index(max(_translate))]

    @property
    def vectorFromParent(self):
        if self.parent:
            return self.worldPosition - self.parent.worldPosition
        else:
            return self.worldPosition

    @property
    def distanceFromParent(self):
        return self.vectorFromParent.length()

    @property
    def shape(self):
        # get the transform's shape if it has one
        try:
            # Don't modify the node's existing dag path object
            _dag = om.MDagPath(self.dagPath)
            # Only supports 1 shape
            _dag = _dag.extendToShape(0)
            # check what the shape's type is
            _shapeMObj = _dag.node()
            if _shapeMObj.hasFn(om.MFn.kMesh):
                _mesh = Mesh(_shapeMObj)
                if not _mesh.fnDagNode.isIntermediateObject:
                    # TODO there are loops happening here that shouldn't be happening
                    return _mesh
            elif _shapeMObj.hasFn(om.MFn.kNurbsCurve):
                _curve = Curve(_shapeMObj)
                if not _curve.fnDagNode.isIntermediateObject:
                    return _curve
            # TODO support more shape types
            else:
                return Shape(_shapeMObj)

        except RuntimeError:
            # no shape
            return None

    def _constrainTo(self, parent, constraintType):
        offsetParentMatrixPlug = self.findPlug('offsetParentMatrix')
        existingPickMatrix = None
        matrixUsageValues = [not bool(matrixPart in CONSTRAINT_TYPE_USAGES[constraintType].keys()) for matrixPart in
                             PICK_MATRIX_PARTS]

        # Check if we're already constrained to this parent
        if self.isNodeUpstream(offsetParentMatrixPlug, parent.mObject):
            existingPickMatrix = self.findUpstreamNodesOfType(om.MFn.kPickMatrix, sourcePlug=offsetParentMatrixPlug)
            if existingPickMatrix:
                existingPickMatrix = existingPickMatrix[0]
                existingPickMatrixUsageValues = [existingPickMatrix.get(matrixPart) for matrixPart in PICK_MATRIX_PARTS]
                if existingPickMatrixUsageValues == matrixUsageValues:
                    om.MGlobal.displayInfo('{0} is already {1} constrained to {2}'.format(self, constraintType, parent))
                    return

        # Create a 'pickMatrix' node and make it only use translation
        if not existingPickMatrix:
            pickMatrixNode = DependencyNode(Scene.createNode('pickMatrix', self.name + '_pickMatrix'))
        else:
            pickMatrixNode = existingPickMatrix

        for matrixPart in PICK_MATRIX_PARTS:
            pickMatrixNode.set(matrixPart, not bool(matrixPart in CONSTRAINT_TYPE_USAGES[constraintType].keys()))

        if existingPickMatrix:
            # make sure it's not connected to something different like an aim matrix node
            pickMatrixInputMatrixPlug = pickMatrixNode.findPlug('inputMatrix')
            sourceConnection = pickMatrixInputMatrixPlug.source()
            if not sourceConnection.isNull:
                pickMatrixNode.disconnect(sourceConnection, pickMatrixInputMatrixPlug)
        else:
            parent.connect(parent.findPlug('matrix'), pickMatrixNode.findPlug('inputMatrix'))
            pickMatrixNode.connect(pickMatrixNode.findPlug('outputMatrix'), self.findPlug('offsetParentMatrix'))

    # TODO: support multi constraints

    def parentConstrainTo(self, parent, maintainOffset=False):
        if maintainOffset:
            _previousPosition = self.worldPosition
            _previousRotation = self.worldRotation  # TODO: maintain scale offset
        
        constraintPickMatrix = ParentConstraint.create(parent, [self])
        constraint = ParentConstraint(constraintPickMatrix.mObject)

        if maintainOffset:
            self.worldPosition = _previousPosition
            self.worldRotation = _previousRotation

        return constraint

    def pointConstrainTo(self, parent, maintainOffset=False):
        if maintainOffset:
            _previousPosition = self.worldPosition
            _previousRotation = self.worldRotation

        self._constrainTo(parent, 'point')

        if maintainOffset:
            self.worldPosition = _previousPosition
            self.worldRotation = _previousRotation

    def orientConstrainTo(self, parent, maintainOffset=False):
        if maintainOffset:
            _previousPosition = self.worldPosition
            _previousRotation = self.worldRotation

        self._constrainTo(parent, 'orient')

        # TODO: this puts the pivots off for the rotation, which gives unnatural results for an orient constraint
        if maintainOffset:
            self.worldPosition = _previousPosition
            self.worldRotation = _previousRotation

    def scaleConstrainTo(self, parent, maintainOffset=False):
        self._constrainTo(parent, 'scale')

    def alignConstrainTo(self, parent, maintainOffset=False):
        # use aimMatrix "align" functionality
        raise NotImplementedError()

    def aimConstrainTo(self, target, aimVector=(1, 0, 0), upVector=(0, 1, 0)):
        offsetParentMatrixPlug = self.findPlug('offsetParentMatrix')
        existingAimMatrix = None
        existingPickMatrix = None
        matrixUsageValues = [not bool(matrixPart in CONSTRAINT_TYPE_USAGES['aim'].keys()) for matrixPart in
                             PICK_MATRIX_PARTS]

        if self.isNodeUpstream(offsetParentMatrixPlug, target.mObject):
            existingPickMatrix = self.findUpstreamNodesOfType(om.MFn.kPickMatrix, sourcePlug=offsetParentMatrixPlug)
            existingPickMatrix = existingPickMatrix[0] if existingPickMatrix else None
            existingAimMatrix = self.findUpstreamNodesOfType(om.MFn.kAimMatrix, sourcePlug=offsetParentMatrixPlug)
            existingAimMatrix = existingAimMatrix[0] if existingAimMatrix else None

        # Check if we're already constrained to this parent
        if self.isNodeUpstream(offsetParentMatrixPlug, target.mObject):
            existingAimMatrix = self.findUpstreamNodesOfType(om.MFn.kAimMatrix, sourcePlug=offsetParentMatrixPlug)
            if existingAimMatrix:
                existingAimMatrix = existingAimMatrix[0]

        if not existingPickMatrix:
            pickMatrixNode = DependencyNode(Scene.createNode('pickMatrix', self.name + '_pickMatrix'))
        else:
            pickMatrixNode = existingPickMatrix

        for matrixPart in PICK_MATRIX_PARTS:
            pickMatrixNode.set(matrixPart, not bool(matrixPart in CONSTRAINT_TYPE_USAGES['aim'].keys()))

        if not existingAimMatrix:
            aimMatrixNode = DependencyNode(Scene.createNode('aimMatrix', self.name + '_aimMatrix'))
        else:
            aimMatrixNode = existingAimMatrix

        if not existingAimMatrix and not existingPickMatrix:
            target.connect(target.findPlug('matrix'), aimMatrixNode.findPlug('inputMatrix'))
            aimMatrixNode.connect(aimMatrixNode.findPlug('outputMatrix'), pickMatrixNode.findPlug('inputMatrix'))
            pickMatrixNode.connect(pickMatrixNode.findPlug('outputMatrix'), self.findPlug('offsetParentMatrix'))

        aimMatrixNode.set('primaryMode', 1)
        aimMatrixNode.set('primaryInputAxisX', aimVector[0])
        aimMatrixNode.set('primaryInputAxisY', aimVector[1])
        aimMatrixNode.set('primaryInputAxisZ', aimVector[2])

        if upVector:
            aimMatrixNode.set('secondaryMode', 1)
            aimMatrixNode.set('secondaryInputAxisX', upVector[0])
            aimMatrixNode.set('secondaryInputAxisY', upVector[1])
            aimMatrixNode.set('secondaryInputAxisZ', upVector[2])


    def lockTransforms(self):
        # This is not undoable
        for attr in ['translate', 'rotate', 'scale']:
            for axis in ['X', 'Y', 'Z']:
                self.lockAttr(attr + axis)

    def unlockTransforms(self):
        # This is not undoable
        for attr in ['translate', 'rotate', 'scale']:
            for axis in ['X', 'Y', 'Z']:
                self.unlockAttr(attr + axis)


class Shape(DagNode):
    def __init__(self, seed):
        super(Shape, self).__init__(seed)

    @property
    def mFnType(self):
        return om.MFn.kShape, 'kShape'

    @property
    def transform(self):
        # return this shape's transform
        _dag = om.MDagPath(self.dagPath)
        return Transform(_dag.transform())


class Mesh(Shape):
    def __init__(self, seed):
        super(Mesh, self).__init__(seed)

    @classmethod
    def getVertexNeighbors(cls, dag, components):
        # also returns itself

        itVerts = om.MItMeshVertex(dag, components)
        neighboringVerts = set()

        while not itVerts.isDone():
            neighboringVerts.add(itVerts.index())
            neighbors = itVerts.getConnectedVertices()
            for neighbor in neighbors:
                neighboringVerts.add(neighbor)
            itVerts.next()

        sel = om.MSelectionList()
        for vertId in neighboringVerts:
            sel.add(dag.fullPathName() + '.vtx[{0}]'.format(vertId))

        dag, neighboringVertices = sel.getComponent(0)

        return neighboringVertices

    @classmethod
    def createFromObjFile(cls, filePath, transformationMatrix=None):
        '''Very simple obj reader for fun'''

        with open(filePath, 'r') as f:
            data = f.readlines()

        # floatMatrix = None

        if transformationMatrix:
            transformationMatrix = om.MMatrix(
                transformationMatrix)  # floatMatrix = om.MFloatMatrix(transformationMatrix)

        vertices = []
        polygonCounts = []
        polygonConnects = []
        uValues = []
        vValues = []
        normals = om.MFloatVectorArray()
        faceIds = []
        vertexIds = []
        faceVertexNormalIndices = []
        uvCounts = []
        uvIds = []
        name = 'polyMesh1'
        missingUVs = False

        faceId = 0

        for line in data:
            line = line.strip('\n')
            if line.startswith('v '):
                # Vertex coordinate
                _point = om.MPoint(*[float(p) for p in line.split(' ')[1:]])
                if transformationMatrix:
                    _point = _point * transformationMatrix
                vertices.append(_point)
            elif line.startswith('vt '):
                # vertex UV coordinate
                _uv = [float(f) for f in line.split(' ')[1:]]
                uValues.append(_uv[0])
                vValues.append(_uv[1])
            elif line.startswith('vn '):
                # vertex normal vector
                # _normal = om.MFloatVector(*[float(n) for n in line.split(' ')[1:]])
                # if floatMatrix:
                #     _normal = _normal * floatMatrix
                _normal = om.MVector(*[float(n) for n in line.split(' ')[1:]])
                if transformationMatrix:
                    _normal = _normal * transformationMatrix
                normals.append(_normal)
            elif line.startswith('g '):
                name = line.split(' ')[1]
            elif line.startswith('f '):
                # Polygon connection information
                # Expecting the format "v/vt/vn"
                # vertex id, vertex uv index, vertex normal index
                _faceVertexData = line.split(' ')[1:]
                numVertsInFace = len(_faceVertexData)
                polygonCounts.append(len(_faceVertexData))
                for faceData in [item.split('/') for item in _faceVertexData]:
                    # OBJ vertex indices start at 1
                    vertId = int(faceData[0]) - 1
                    polygonConnects.append(vertId)
                    faceIds.append(faceId)
                    vertexIds.append(vertId)
                    faceVertexNormalIndices.append(int(faceData[2]) - 1)

                    if faceData[1] and not missingUVs:
                        uvIds.append(int(faceData[1]) - 1)
                    else:
                        missingUVs = True

                uvCounts.append(numVertsInFace)

                faceId += 1

        fnMesh = om.MFnMesh()
        meshMObject = fnMesh.create(vertices, polygonCounts, polygonConnects)

        if not missingUVs:
            fnMesh.setUVs(uValues, vValues)
            fnMesh.assignUVs(uvCounts, uvIds)

        # print(normals)
        # print(len(normals))
        # print(faceIds)
        # print(len(faceIds))
        # print(vertexIds)
        # print(len(vertexIds))
        # print(len(vertices))
        # print(sum(polygonCounts))

        if len(normals) == sum(polygonCounts):
            # completely hard shaded
            fnMesh.setFaceVertexNormals(normals, faceIds, vertexIds)
        elif len(normals) == len(vertices):
            # completely smooth shaded
            fnMesh.setNormals(normals)
        else:
            # We'll have to do more work if the object has a mixture of smooth and hard shaded edges
            faceVertexNormals = []
            for i in xrange(len(faceIds)):
                faceVertexNormals.append(normals[vertexIds[i]])

            fnMesh.setFaceVertexNormals(faceVertexNormals, faceIds, vertexIds)

        DG_MODIFIER.renameNode(meshMObject, name)
        DG_MODIFIER.doIt()

        return meshMObject

    @property
    def mFnType(self):
        return om.MFn.kMesh, 'kMesh'

    def getVertexColors(self):
        if self.fnMesh.numColorSets:
            return self.fnMesh.getVertexColors(self.fnMesh.currentColorSetName())
        else:
            return None

    def setVertexColors(self, colors, vertexIds):
        if self.fnMesh.numColorSets:
            self.fnMesh.setVertexColors(colors, vertexIds, DAG_MODIFIER)

        DAG_MODIFIER.doIt()

    def projectOnto(self, targetMesh, projectFromNormal=False):
        '''Returns the vertex id map and the corresponding position mapping between the own and target mesh.
        Does a world space closestPoint OR closestIntersection projection of the target mesh onto the own mesh.
        '''

        # get the ownMesh's vertex positions
        targetFnMesh = targetMesh.fnMesh
        ownFnMesh = self.fnMesh
        ownMeshVertexPositions = targetFnMesh.getPoints(om.MSpace.kWorld)
        faceVertexCounts, meshRelativeFaceIds = targetFnMesh.getVertices()
        ownMeshIter = om.MItMeshVertex(self.dagPath)
        hasMissedProjections = False

        # OwnVertexIndex >>> ClosestTargetVertexIndex
        closestVertexMapping = {}
        # OwnVertexPosition >>> ClosestTargetVertexPosition
        vertexPositionsMapping = []
        closestPointMapping = {}

        while not ownMeshIter.isDone():
            ownVertexIndex = int(ownMeshIter.index())
            # get the closest point on the own mesh to this targetMesh vertex
            vertexPosition = ownMeshIter.position(om.MSpace.kWorld)

            if projectFromNormal:
                raySource = om.MFloatPoint(vertexPosition.x, vertexPosition.y, vertexPosition.z)
                # get the angle weighted vertex normal
                vertexNormal = ownFnMesh.getVertexNormal(ownVertexIndex, True, om.MSpace.kWorld)
                rayDirection = om.MFloatVector(vertexNormal.x, vertexNormal.y, vertexNormal.z)
                maxDistance = 1.0
                testBothDirections = True
                # returns closestPoint, hitRayParam, closestFaceIndex, hitTriangle, hitBary1, hitBary2
                hitResults = targetFnMesh.closestIntersection(raySource, rayDirection, om.MSpace.kWorld, maxDistance,
                                                              testBothDirections)

                if hitResults[0] == ORIGIN_POINT_FLOAT and hitResults[1] == 0.0:
                    # No hit
                    hasMissedProjections = True
                    closestPoint, closestFaceIndex = targetFnMesh.getClosestPoint(vertexPosition)
                else:
                    closestPoint = om.MPoint(hitResults[0].x, hitResults[0].y, hitResults[0].z)
                    closestFaceIndex = hitResults[2]

            else:
                closestPoint, closestFaceIndex = targetFnMesh.getClosestPoint(vertexPosition)

            closestDistance = INFINITY
            closestVertexId = 0
            numFaceVertices = faceVertexCounts[closestFaceIndex]
            closestVertexPosition = ORIGIN_POINT
            faceVertexIds = targetFnMesh.getPolygonVertices(closestFaceIndex)

            closestPointVector = om.MVector(closestPoint.x, closestPoint.y, closestPoint.z)

            for i in xrange(numFaceVertices):
                # mesh-relative face id
                faceVertexId = faceVertexIds[i]
                faceVertexPosition = ownMeshVertexPositions[faceVertexId]
                faceVertexVector = om.MVector(faceVertexPosition.x, faceVertexPosition.y, faceVertexPosition.z)
                hitPointToVertexPositionDistance = (closestPointVector - faceVertexVector).length()

                if hitPointToVertexPositionDistance < closestDistance:
                    closestVertexId = faceVertexId
                    closestDistance = hitPointToVertexPositionDistance
                    closestVertexPosition = faceVertexPosition

            closestVertexMapping[ownVertexIndex] = int(closestVertexId)
            vertexPositionsMapping.append((vertexPosition, closestVertexPosition))
            closestPointMapping[ownVertexIndex] = closestPoint

            ownMeshIter.next()

        targetFnMesh.freeCachedIntersectionAccelerator()

        if hasMissedProjections:
            om.MGlobal.displayWarning('Some projections had no hit and the closest point was used as a backup.')

        return closestVertexMapping, vertexPositionsMapping, closestPointMapping

    def isInside(self, targetMesh):
        '''Checks if any point on this mesh is inside the given mesh. If they are, it colors them Red'''

        # get the normals on the target mesh
        targetFnMesh = targetMesh.fnMesh
        targetMeshNormals = targetFnMesh.getNormals(om.MSpace.kWorld)

        ownFaceIter = om.MItMeshPolygon(self.dagPath)

        facesInsideMesh = []

        while not ownFaceIter.isDone():
            facePosition = ownFaceIter.center(om.MSpace.kWorld)
            faceNormal = ownFaceIter.getNormal(om.MSpace.kWorld)

            raySource = om.MFloatPoint(facePosition.x, facePosition.y, facePosition.z)
            rayDirection = om.MFloatVector(faceNormal.x, faceNormal.y, faceNormal.z)
            maxDistance = 1.0
            testBothDirections = False
            # returns closestPoint, hitRayParam, closestFaceIndex, hitTriangle, hitBary1, hitBary2
            hitResults = targetFnMesh.closestIntersection(raySource, rayDirection, om.MSpace.kWorld, maxDistance,
                                                          testBothDirections)

            if hitResults[0] == ORIGIN_POINT_FLOAT and hitResults[1] == 0.0:
                # no hit. It's definitely outside the mesh
                ownFaceIter.next()
            else:
                closestPoint = hitResults[0]
                closestFaceIndex = hitResults[2]

                normalSum = om.MFloatVector()
                faceVertexIndices = targetFnMesh.getPolygonVertices(closestFaceIndex)
                numFaceVertices = len(faceVertexIndices)

                for i in xrange(numFaceVertices):
                    # average the face's normals
                    normalSum += targetMeshNormals[faceVertexIndices[i]]

                faceNormal = (normalSum / float(numFaceVertices)).normal()

                vertexPositionVector = om.MFloatVector(facePosition.x, facePosition.y, facePosition.z)
                closestPointVector = om.MFloatVector(closestPoint.x, closestPoint.y, closestPoint.z)

                vertexPositionToClosestPointVector = closestPointVector - vertexPositionVector

                dot = vertexPositionToClosestPointVector * faceNormal

                if dot > 0.0:
                    # inside the mesh
                    facesInsideMesh.append(int(ownFaceIter.index()))

                ownFaceIter.next()

        targetFnMesh.freeCachedIntersectionAccelerator()

        if len(facesInsideMesh):
            return True, facesInsideMesh
        else:
            return False, facesInsideMesh

    @property
    def fnMesh(self):
        try:
            # world space operations only work if the fnMesh was created from a dag path
            _fnMesh = om.MFnMesh(self.dagPath)
            return _fnMesh
        except RuntimeError:
            raise RuntimeError('Given object type is not compatible. Is the object a polygonal mesh?')

    def getSkinCluster(self):
        _upstreamSkinClusters = self.findUpstreamNodesOfType(om.MFn.kSkinClusterFilter)
        if not _upstreamSkinClusters:
            return None
        else:
            return SkinCluster(_upstreamSkinClusters[0].mObject)

    def getShaders(self):
        '''Returns the per-polygon shader assignment for this instance of the mesh'''

        shaderGroupMObjects, polygonToShaderMapping = self.fnMesh.getConnectedShaders(0)

        shaderGroups = []
        shaders = []

        for shaderMObject in shaderGroupMObjects:
            if shaderMObject.apiType() == om.MFn.kShadingEngine:
                shaderGroup = ShaderGroup(shaderMObject)
                shaderGroups.append(shaderGroup)

        for shaderGroup in shaderGroups:
            shaders.append(shaderGroup.material)

        return shaders, polygonToShaderMapping

    def assignMaterial(self, material):
        # TODO: create a Material node type
        instObjectGroupsPlug = self.findPlug('instObjGroups')
        instObjectGroupsPlug = instObjectGroupsPlug.elementByLogicalIndex(0)
        if instObjectGroupsPlug.isConnected:
            # Destinations skips over unit conversion nodes. Neat
            connectedPlugs = instObjectGroupsPlug.destinations()
            for plug in connectedPlugs:
                shaderGroup = plug.node()
                shaderGroupSet = om.MFnSet(shaderGroup)
                if shaderGroupSet.isMember(self.mObject):
                    shaderGroupSet.removeMember(self.mObject)

        material = Shader(material)
        shaderGroupMObject = material.shaderGroup

        if not shaderGroupMObject:
            raise RuntimeError('The requested material {0} is not connected to a shader group.'.format(material))

        shadingGroupSet = om.MFnSet(shaderGroupMObject)
        shadingGroupSet.addMember(self.mObject)


class Curve(DagNode):
    def __init__(self, seed):
        super(Curve, self).__init__(seed)

    @classmethod
    def fitCurveToPoints(cls, points, name='fitCurve'):
        # TODO: do this with MDGModifier
        linearCurve = LinearCurve(LinearCurve.create(points=points))
        fitCurve = cmds.fitBspline(linearCurve, constructionHistory=False, tolerance=0.001, name=name)[0]
        cmds.delete(linearCurve)

        return Curve(fitCurve)

    @property
    def mFnType(self):
        return om.MFn.kNurbsCurve, 'kNurbsCurve'

    @property
    def fnNurbsCurve(self):
        return om.MFnNurbsCurve(self.dagPath)

    def getPoints(self, space='world'):
        space = SPACES.get(space)
        if not space:
            raise ValueError('Unsupported space type \'{0}\' given.'.format(space))

        _cvs = self.fnNurbsCurve.cvPositions(space)

        return _cvs

    def setPoints(self, points, space='world', cvLimit=128):
        space = SPACES.get(space)
        if not space:
            raise ValueError('Unsupported space type \'{0}\' given.'.format(space))

        _points = om.MPointArray()
        _numCVs = self.fnNurbsCurve.numCVs

        if isinstance(points, list):
            for i in xrange(len(points)):
                _points.append(om.MPoint(*points[i]))
            _lastPoint = _points[len(points) - 1]
            for j in xrange(len(points), cvLimit):
                _points.append(om.MPoint(_lastPoint))

        elif isinstance(points, om.MPointArray):
            _points = points  # TODO: this isn't fully supported yet  # if points.length() != _numCVs:  #     raise ValueError('The number of points given does not match the number of CV points.')

        self.fnNurbsCurve.setCVPositions(_points, space)
        self.fnNurbsCurve.updateCurve()

    def setupBoundingBox(self):

        sceneBoundingBox = Scene.getBoundingBox()

        if sceneBoundingBox.width() > 0.001:
            cmds.xform(self.shape + '.cv[0:]', scale=(
                sceneBoundingBox.width() + 0.1, sceneBoundingBox.height() + 0.1, sceneBoundingBox.depth() + 0.1))

        center = sceneBoundingBox.center()
        cmds.xform(self.shape + '.cv[0:]', translation=(center.x, center.y, center.z), relative=True)


class LinearCurve(Curve):
    def __init__(self, seed):
        super(LinearCurve, self).__init__(seed)

    @property
    def mFnType(self):
        return om.MFn.kNurbsCurve, 'kNurbsCurve'

    @classmethod
    def create(cls, points=(), forceCvCount=False):
        om.MGlobal.clearSelectionList()
        _points = om.MPointArray()
        _knots = om.MDoubleArray()

        if not isinstance(points, om.MPointArray):
            for i in xrange(len(points)):
                _points.append(om.MPoint(*points[i]))
                _knots.append(i)

            if forceCvCount:
                # Force the cv count to be the same on rig control curves so their shape can be easily changed
                _lastPoint = _points[len(_points) - 1]
                for j in xrange(len(points), forceCvCount):
                    # make all the remaining CVs equal to the same as the last point
                    _points.append(om.MPoint(_lastPoint))
                    _knots.append(j)
        else:
            _points = points

        _newCurveMObj = om.MFnNurbsCurve().create(_points, _knots, 1, om.MFnNurbsCurve.kOpen, False, False)

        return _newCurveMObj

    def attachBorderCVsToTransform(self, startTransform, endTransform):
        if not len(self.getPoints()) == 0:
            raise RuntimeError('This function can only be run on a curve with 2 points (a line).')

        # decompA = cmds.shadingNode('decomposeMatrix', asUtility=True, name=name + '_decompMatrixA')
        # decompB = cmds.shadingNode('decomposeMatrix', asUtility=True, name=name + '_decompMatrixB')
        # cmds.connectAttr(pointA + '.worldMatrix[0]', decompA + '.inputMatrix')
        # cmds.connectAttr(pointB + '.worldMatrix[0]', decompB + '.inputMatrix')
        #
        # cmds.connectAttr(decompA + '.outputTranslate', crvShape + '.controlPoints[0]')
        # cmds.connectAttr(decompB + '.outputTranslate', crvShape + '.controlPoints[1]')
        #
        # cmds.parent(crv, parentNode, a=1)
        # cmds.setAttr(crv + '.inheritsTransform', 0)
        # cmds.setAttr(crvShape + '.template', 1)
        #
        # cmds.xform(crv, translation=(0, 0, 0), a=1, ws=1)
        # cmds.xform(crv, rotation=(0, 0, 0), a=1, ws=1)
        #
        # cmds.select(cl=1)

        return


class SkinCluster(DependencyNode):
    def __init__(self, seed):
        super(SkinCluster, self).__init__(seed)

    @classmethod
    def pruneVertexWeightList(cls, weights, influenceCount, maxInfluence=4, normalize=True):
        """Prunes the given list to have only four non-zero values"""

        sortedWeights = weights
        sortedWeights.sort(reverse=True)
        prunedWeights = [0.0] * influenceCount

        if maxInfluence > 1:
            for i in xrange(influenceCount):
                if i > (maxInfluence - 1):
                    prunedWeights[weights.index(sortedWeights[i])] = 0.0
                else:
                    prunedWeights[weights.index(sortedWeights[i])] = sortedWeights[i]
        else:
            prunedWeights[weights.index(sortedWeights[0])] = sortedWeights[0]

        if normalize:
            prunedWeightsSum = sum(prunedWeights)
            prunedWeights = [value / prunedWeightsSum for value in prunedWeights]

        return prunedWeights

    @property
    def fnSkinCluster(self):
        return omAnim.MFnSkinCluster(self.mObject)

    @property
    def outputGeometry(self):
        outputGeometryPlug = self.findPlug('outputGeometry')
        # outputGeometryPlug = outputGeometryArrayPlug.elementByLogicalIndex(0)
        connectedMeshes = outputGeometryPlug.connectedTo(False, True)
        # return the first connected mesh
        return Mesh(connectedMeshes[0].node())

    @property
    def influenceObjects(self):
        _influenceObjects = self.fnSkinCluster.influenceObjects()
        return _influenceObjects

    @property
    def influenceCount(self):
        return len(self.influenceObjects)

    @property
    def infCountPointer(self):
        _infCount = om.MScriptUtil(self.influenceCount)
        # c++ utility needed for the get weights function
        _infCountPtr = _infCount.asUintPtr()
        return _infCountPtr

    @property
    def influenceIndices(self):
        _influenceIndices = om.MIntArray()
        # create an MIntArray that just counts from 0 to infCount
        for i in xrange(self.influenceCount):
            _influenceIndices.append(i)

        return _influenceIndices

    def getWeights(self, components=None):
        _weights = om.MDoubleArray()

        if not components:
            components = om.MObject()
        else:
            assert isinstance(components, om.MObject)

        _outputGeometryDag = self.outputGeometry.dagPath
        _influenceIndices = self.influenceIndices

        _weights = self.fnSkinCluster.getWeights(_outputGeometryDag, components, _influenceIndices)

        return _weights

    def getWeightDictionary(self):
        '''Return the weight dictionary for the given skin cluster.
        Each influence's short name is followed by its vertex weights.'''

        weightDict = {}

        _weights = self.getWeights()
        _weights = list(_weights)
        _influenceIndices = self.influenceIndices
        _influenceObjects = self.influenceObjects

        numInfluences = self.influenceCount

        for i in xrange(numInfluences):
            # A single influence's weights is its column of the weight matrix
            weightDict[_influenceObjects[i].partialPathName()] = _weights[i:: numInfluences]

        return weightDict

    def setWeights(self, newWeights, components=None):
        if not components:
            components = NULL_OBJECT
        else:
            assert isinstance(components, om.MObject)

        if not isinstance(newWeights, om.MDoubleArray):
            if isinstance(newWeights, list):
                _temp = om.MDoubleArray()
                for w in newWeights:
                    _temp.append(w)
                newWeights = _temp
            else:
                raise ValueError('newWeights must be a DoubleArray or list')

        self.fnSkinCluster.setWeights(self.outputGeometry.dagPath, components, self.influenceIndices, newWeights)

    def pruneWeights(self, maxInfluence=4):
        """Sets the weights for each vertex to the average of its neighbors."""

        originalSel = om.MGlobal.getActiveSelectionList()

        if not len(originalSel):
            cmds.select(self.outputGeometry.dagPath.fullPathName())

        _dag, selectedComponents = Scene.getSelectedVertexIDs()

        oldWeights = self.getWeights(components=selectedComponents)

        # newWeights just starts as a copy of oldWeights
        newWeights = om.MDoubleArray(oldWeights)

        # iterate over the selected verts
        itVerts = om.MItMeshVertex(_dag, selectedComponents)
        i = 0
        numOverInfluenced = 0
        infCount = self.influenceCount

        while not itVerts.isDone():
            vertWeightSliceStart = i * infCount
            vertWeights = list(newWeights[vertWeightSliceStart: vertWeightSliceStart + infCount])
            # Filter is nearly instant.
            nonZeroCount = len(filter(lambda w: w > 0.0, vertWeights))

            if nonZeroCount > maxInfluence:
                numOverInfluenced += 1
                # makes the weights for the closest vertex equal to the outer vertex
                newWeights[vertWeightSliceStart: vertWeightSliceStart + infCount] = self.pruneVertexWeightList(
                    vertWeights, influenceCount=infCount, maxInfluence=maxInfluence)

            i += 1
            itVerts.next()

        if numOverInfluenced:
            self.setWeights(newWeights, components=selectedComponents)

        om.MGlobal.setActiveSelectionList(originalSel)

        om.MGlobal.displayInfo(
            'Pruned {0} over-influenced vertices from {1}.'.format(numOverInfluenced, _dag.partialPathName()))

    def smoothWeights(self, iterations=3, pinBorderVerts=False):
        oldWeights = self.getWeights()
        newWeights = oldWeights

        _dag, selectedComponents = Scene.getSelectedVertexIDs()
        numInfluences = self.influenceCount

        for iteration in xrange(iterations):
            # TODO: find a way to do this more efficiently, maybe be precaching neighbor verts
            # newWeights just starts as a copy of oldWeights
            newWeights = list(om.MDoubleArray(oldWeights))

            # iterate over the selected verts
            itVerts = om.MItMeshVertex(_dag, selectedComponents)

            while not itVerts.isDone():
                if pinBorderVerts and itVerts.onBoundary():
                    itVerts.next()
                else:
                    neighborVerts = itVerts.getConnectedVertices()
                    neighborWeightSums = [0.0] * numInfluences
                    numNeighbors = len(neighborVerts)
                    vertId = itVerts.index()

                    # TODO: see if there's a way to get rid of this nested for loop
                    for i in xrange(numNeighbors):
                        v = neighborVerts[i]
                        # get these vertex's weights and add them to our weight sums
                        for j in xrange(numInfluences):
                            neighborWeightSums[j] += oldWeights[(v * numInfluences) + j]

                    smoothedWeight = [w / float(numNeighbors) for w in neighborWeightSums]
                    weightSum = float(sum(smoothedWeight))
                    normalizedWeight = [w / float(weightSum) for w in smoothedWeight]

                    newWeights[(vertId * numInfluences): (vertId * numInfluences) + numInfluences] = normalizedWeight

                    itVerts.next()

            oldWeights = newWeights

        self.setWeights(newWeights)


class ShaderGroup(DependencyNode):
    def __init__(self, seed):
        super(ShaderGroup, self).__init__(seed)

        self.fnSet = om.MFnSet(self.mObject)

    @property
    def material(self):
        surfaceShaderPlug = self.findPlug('surfaceShader')
        inputPlug = surfaceShaderPlug.source()
        # TODO: do type check
        return Shader(inputPlug.node())


class Shader(DependencyNode):
    blinnTextureSlotNames = {'albedo': 'color', 'alpha': 'transparency', 'roughness': 'eccentricity',
                             'metalness': 'reflectivity'}
    arnoldTextureSlotNames = {'albedo': 'baseColor', 'alpha': 'opacity', 'roughness': 'specularRoughness'}

    def __init__(self, seed):
        super(Shader, self).__init__(seed)

    @property
    def textures(self):
        _textures = {}
        for textureType in ['albedo', 'specular', 'alpha', 'normal', 'roughness', 'metalness']:
            _textures[textureType] = getattr(self, textureType + 'Texture')
        return _textures

    @property
    def shaderGroup(self):
        '''Returns the MObject of the material's shader group'''

        connectedPlugs = self.findPlug('outColor').destinations()
        shaderGroupMObject = None

        for plug in connectedPlugs:
            node = plug.node()
            if node.apiType() == om.MFn.kShadingEngine:
                # A material can only belong to one shader group, so we can break as soon as we find one.
                shaderGroupMObject = node
                break

        return ShaderGroup(shaderGroupMObject)

    @property
    def isArnoldShader(self):
        result = False
        pluginName = self.fnDependencyNode.pluginName
        if pluginName and pluginName.endswith('mtoa.mll'):
            result = True

        return result

    @property
    def albedoTexture(self):
        return self.getSimpleTexture(self.getTextureSlotName('albedo'))

    @property
    def specularTexture(self):
        return self.getSimpleTexture(self.getTextureSlotName('specularColor'))

    @property
    def alphaTexture(self):
        return self.getSimpleTexture(self.getTextureSlotName('alpha'))

    @property
    def normalTexture(self):
        textureSlotName = self.getTextureSlotName('normalCamera')
        bump2dNode = self.getInputConnection(textureSlotName)
        if not bump2dNode:
            return None

        texturePath = bump2dNode.getFileTexturePath('bumpValue')

        return texturePath

    @property
    def roughnessTexture(self):
        return self.getSimpleTexture(self.getTextureSlotName('roughness'))

    @property
    def metalnessTexture(self):
        return self.getSimpleTexture(self.getTextureSlotName('metalness'))

    def getConnectedShapes(self):
        # get the members of the shader group. Polygonal members will return the shape
        # return a flattened list of the shader group set's members
        memberList = self.shaderGroup.fnSet.getMembers(True)
        shapes = []

        itSel = om.MItSelectionList(memberList)

        while not itSel.isDone():
            shapes.append(Shape(itSel.getDependNode()))
            itSel.next()

        return shapes

    def getTextureSlotName(self, textureType):
        isArnold = self.isArnoldShader

        if isArnold:
            if textureType in self.arnoldTextureSlotNames.keys():
                return self.arnoldTextureSlotNames[textureType]
            else:
                return textureType
        else:
            if textureType in self.blinnTextureSlotNames.keys():
                return self.blinnTextureSlotNames[textureType]
            else:
                return textureType

    def getSimpleTexture(self, textureSlotName):
        texturePath = self.getFileTexturePath(textureSlotName)

        if texturePath:
            return texturePath
        else:
            return self.get(textureSlotName)


class Constraint(DependencyNode):
    def __init__(self, seed):
        super(Constraint, self).__init__(seed)

        self.initialized = False
        self.parent = None
        self.children = None
        self.pickMatrix = None

    @property
    def constraintType(self):
        raise NotImplementedError()

    def create(self, parent, children):
        raise NotImplementedError()


class ParentConstraint(Constraint):
    # TODO: improve the process of creating a constraint
    '''
    constraint = ParentConstraint(parent, child/children)
    - would need to be able to create a ParentConstraint from its init (so no seed)
    - OR the syntax could stay: constraint = parent.parentConstrainTo(child/children)
      - that fn could take care of the object's creation.

    '''
    def __init__(self, seed):
        super(ParentConstraint, self).__init__(seed)

    @property
    def constraintType(self):
        return 'parent'

    @classmethod
    def create(cls, parent, children):
        # TODO: type checks
        if isinstance(children, Transform):
            children = [children]

        for child in children:
            offsetParentMatrixPlug = child.findPlug('offsetParentMatrix')
            existingPickMatrix = None

            # Check if we're already constrained to this parent
            if child.isNodeUpstream(offsetParentMatrixPlug, parent.mObject):
                existingPickMatrixNodes = child.findUpstreamNodesOfType(om.MFn.kPickMatrix,
                                                                        sourcePlug=offsetParentMatrixPlug)
                if existingPickMatrix:
                    # Make sure the DG Modifier's queue is empty before calling delete node.
                    for pickMatrixNode in existingPickMatrixNodes:
                        DG_MODIFIER.doIt()
                        DG_MODIFIER.deleteNode(pickMatrixNode.mObject)
                        DG_MODIFIER.doIt()

            pickMatrix = DependencyNode(Scene.createNode('pickMatrix', child + '_pickMatrix'))

            for matrixPart in PICK_MATRIX_PARTS:
                pickMatrix.set(matrixPart, not bool(matrixPart in CONSTRAINT_TYPE_USAGES['parent'].keys()))
            else:
                parent.connect(parent.findPlug('matrix'), pickMatrix.findPlug('inputMatrix'))
                pickMatrix.connect(pickMatrix.findPlug('outputMatrix'), child.findPlug('offsetParentMatrix'))

        # TODO: return type... return only one pick matrix node for all children?
        return pickMatrix