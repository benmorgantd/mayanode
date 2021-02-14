import sys

import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as omAnim

def maya_useNewAPI():
    pass


class SmoothSkinWeights(om.MPxCommand):
    COMMAND_NAME = 'bmSmoothSkinWeights'
    undoQueue = []
    dagPath = om.MDagPath()
    selectedComponents = om.MObject()
    selectedComponentsWithNeighbors = om.MObject()
    pinBorderVerts = False
    vertexId = None
    strength = 1.0
    # pruneWeightsBelow = 0.005

    def __init__(self):
        super(SmoothSkinWeights, self).__init__()

    @classmethod
    def isUndoable(cls):
        return True

    @classmethod
    def hasSyntax(cls):
        return True

    @classmethod
    def syntax(cls):
        _syntax = om.MSyntax()
        # optionally pass the number of iterations
        # _syntax.addFlag('i', 'iterations', om.MSyntax.kLong)
        _syntax.addFlag('pin', 'pinBorderVerts', om.MSyntax.kBoolean)
        # _syntax.addFlag('pwb', 'pruneWeightsBelow', om.MSyntax.kDouble)
        _syntax.addFlag('v', 'vertexId', om.MSyntax.kLong)
        _syntax.addFlag('s', 'strength', om.MSyntax.kDouble)

        return _syntax

    @classmethod
    def getVertexNeighbors(cls, dag, components):
        # also returns itself

        itVerts = om.MItMeshVertex(dag, components)
        neighboringVerts = set()

        while not itVerts.isDone():
            neighboringVerts.add(itVerts.index())
            neighbors = itVerts.getConnectedVertices()
            neighboringVerts.update(neighbors)
            itVerts.next()

        sel = om.MSelectionList()
        for vertId in neighboringVerts:
            sel.add(dag.fullPathName() + '.vtx[{0}]'.format(vertId))

        dag, neighboringVertices = sel.getComponent(0)

        return neighboringVertices

    @classmethod
    def getVertexComponentFromArg(cls, vertexId):
        sel = om.MGlobal.getActiveSelectionList()
        dag = sel.getDagPath(0)

        component = None

        # check what the shape's type is
        # TODO: put this in a try/except if the transform has no shape
        dag.extendToShape(0)
        shapeMObj = dag.node()
        if shapeMObj.hasFn(om.MFn.kMesh):
            meshMFnDagNode = om.MFnDagNode(shapeMObj)
            if not meshMFnDagNode.isIntermediateObject:
                component = dag.fullPathName() + '.vtx[{0}]'.format(vertexId)

        newSel = om.MSelectionList()
        newSel.add(component)

        dag, component = newSel.getComponent(0)
        return dag, component

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
                meshMFnDagNode = om.MFnDagNode(_shapeMObj)
                meshMFn = om.MFnMesh(_shapeMObj)
                if not meshMFnDagNode.isIntermediateObject:
                    vertexConversion.update(xrange(meshMFn.numVertices))
            else:
                return

        newSel = om.MSelectionList()
        for vertId in vertexConversion:
            newSel.add(dag.fullPathName() + '.vtx[{0}]'.format(vertId))

        dag, selectedComponents = newSel.getComponent(0)

        return dag, selectedComponents

    @classmethod
    def findUpstreamNodesOfType(cls, sourceMObject, nodeType, sourcePlug=None):
        itDG = om.MItDependencyGraph(sourceMObject)
        if sourcePlug:
            itDG.resetTo(sourcePlug, nodeType, om.MItDependencyGraph.kUpstream, om.MItDependencyGraph.kDepthFirst,
                         om.MItDependencyGraph.kNodeLevel)
        else:
            itDG.resetTo(sourceMObject, nodeType, om.MItDependencyGraph.kUpstream, om.MItDependencyGraph.kDepthFirst,
                         om.MItDependencyGraph.kNodeLevel)

        if itDG.isDone():
            # No nodes upstream of that plug of the given type
            return None

        nodeMObjects = []

        while not itDG.isDone():
            nodeMObjects.append(itDG.currentNode())
            itDG.next()

        return nodeMObjects

    def doIt(self, args):
        # gather the values from args
        argParser = om.MArgParser(self.syntax(), args)

        if argParser.isFlagSet('pinBorderVerts'):
            self.pinBorderVerts = argParser.flagArgumentBool('pinBorderVerts', 0)
        if argParser.isFlagSet('pin'):
            self.pinBorderVerts = argParser.flagArgumentBool('pin', 0)

        if argParser.isFlagSet('vertexId'):
            self.vertexId = argParser.flagArgumentInt('vertexId', 0)
        if argParser.isFlagSet('v'):
            self.vertexId = argParser.flagArgumentInt('v', 0)

        if argParser.isFlagSet('strength'):
            self.strength = argParser.flagArgumentDouble('strength', 0)
        if argParser.isFlagSet('s'):
            self.strength = argParser.flagArgumentDouble('s', 0)

        # if argParser.isFlagSet('pruneWeightsBelow'):
        #     self.pruneWeightsBelow = args.flagArgumentDouble('pruneWeightsBelow', 0)
        # if argParser.isFlagSet('pwb'):
        #     self.pruneWeightsBelow = args.flagArgumentDouble('pwb', 0)

        # get the selected components
        if self.vertexId is None:
            self.dagPath, self.selectedComponents = self.getSelectedVertexIDs()
        else:
            self.dagPath, self.selectedComponents = self.getVertexComponentFromArg(self.vertexId)

        self.selectedComponentsWithNeighbors = self.getVertexNeighbors(self.dagPath, self.selectedComponents)

        if not self.dagPath.node().hasFn(om.MFn.kMesh):
            om.MGlobal.displayError('The given object {} is not a mesh shape.'.format(self.dagPath.partialPathName()))
            return

        self.redoIt()

    def redoIt(self):

        # get the skin cluster MObject by going up the history
        skinClusterMObjects = self.findUpstreamNodesOfType(self.dagPath.node(), om.MFn.kSkinClusterFilter)
        if not skinClusterMObjects:
            om.MGlobal.displayError('Given mesh {} has no skin cluster.'.format(self.dagPath.partialPathName()))
            return

        skinClusterMObject = skinClusterMObjects[0]
        fnSkinCluster = omAnim.MFnSkinCluster(skinClusterMObject)

        numInfluences = len(fnSkinCluster.influenceObjects())
        influenceIndices = om.MIntArray()
        for influence in xrange(numInfluences):
            influenceIndices.append(influence)

        # for the starting weights, we need to get weights on every vertex so we have neighbors
        # originalWeights = fnSkinCluster.getWeights(self.dagPath, emptyComponent, influenceIndices)
        # make the original weights just the selected components and its neighbor weights
        originalWeights = fnSkinCluster.getWeights(self.dagPath, self.selectedComponentsWithNeighbors, influenceIndices)
        oldWeights = om.MDoubleArray(originalWeights)

        vertIdToWeightListIndexMap = {}
        # Quickly create a mapping between the weights and their vertex id
        itVerts = om.MItMeshVertex(self.dagPath, self.selectedComponentsWithNeighbors)

        weightListId = 0
        while not itVerts.isDone():
            vertIdToWeightListIndexMap[int(itVerts.index())] = weightListId
            weightListId += 1

            itVerts.next()

        # newWeights just starts as a copy of oldWeights
        newWeights = list(om.MDoubleArray(oldWeights))

        # iterate over the selected verts
        itVerts = om.MItMeshVertex(self.dagPath, self.selectedComponents)

        weightListId = 0

        while not itVerts.isDone():
            isBoundaryVertex = itVerts.onBoundary()
            # option to skip evaluation on boundary vertices
            if self.pinBorderVerts and isBoundaryVertex:
                itVerts.next()
            else:
                neighborVerts = itVerts.getConnectedVertices()
                neighborWeightSums = [0.0] * numInfluences
                numNeighbors = len(neighborVerts)
                # When trying to smooth weights on just a subset of vertices, we need a way to
                # match a vertex's neighbor vert id to that neighbor vertex's weight list index.
                for i in xrange(numNeighbors):
                    neighborVertexIndex = neighborVerts[i]
                    neighborWeightIndex = vertIdToWeightListIndexMap[neighborVertexIndex]
                    # get these vertex's weights and add them to our weight sums
                    for j in xrange(numInfluences):
                        neighborWeightSums[j] += oldWeights[(neighborWeightIndex * numInfluences) + j]

                smoothedWeights = [w / float(numNeighbors) for w in neighborWeightSums]
                strengthWeightedWeights = []

                for weightIndex in xrange(len(smoothedWeights)):
                    smoothedWeight = smoothedWeights[weightIndex]
                    oldWeight = oldWeights[weightIndex]
                    strengthWeightedWeights.append(oldWeight + (self.strength * (smoothedWeight - oldWeight)))

                weightSum = float(sum(strengthWeightedWeights))
                normalizedWeights = [w / float(weightSum) for w in strengthWeightedWeights]

                newWeights[(weightListId * numInfluences): (weightListId * numInfluences) + numInfluences] = normalizedWeights

                weightListId += 1

                itVerts.next()

        newWeightsDoubleArray = om.MDoubleArray()
        for weightIndex in xrange(len(newWeights)):
            # make the final value weighted
            newWeightsDoubleArray.append(newWeights[weightIndex])

        fnSkinCluster.setWeights(self.dagPath, self.selectedComponents, influenceIndices, newWeightsDoubleArray)

        self.undoQueue.append((skinClusterMObject, originalWeights, self.dagPath, influenceIndices, self.selectedComponentsWithNeighbors))

    def undoIt(self):
        if self.undoQueue:
            fnSkinCluster = omAnim.MFnSkinCluster(self.undoQueue[-1][0])
            oldWeights = self.undoQueue[-1][1]
            meshDagPath = self.undoQueue[-1][2]
            influenceIndices = self.undoQueue[-1][3]
            components = self.undoQueue[-1][4]
            # when undoing, we have to set weights on every component
            fnSkinCluster.setWeights(meshDagPath, components, influenceIndices, oldWeights)

            self.undoQueue.pop(-1)


def cmdCreator():
    return SmoothSkinWeights()


def initializePlugin(mObject):
    mPlugin = om.MFnPlugin(mObject)
    try:
        mPlugin.registerCommand(SmoothSkinWeights.COMMAND_NAME, cmdCreator)
    except:
        sys.stderr.write('Failed to register command {}'.format(SmoothSkinWeights.COMMAND_NAME))


def uninitializePlugin(mObject):
    mPlugin = om.MFnPlugin(mObject)
    try:
        mPlugin.deregisterCommand(SmoothSkinWeights.COMMAND_NAME)
    except:
        sys.stderr.write('Failed to unregister command {}'.format(SmoothSkinWeights.COMMAND_NAME))
