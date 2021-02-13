import sys

import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as omAnim

def maya_useNewAPI():
    pass


class SmoothSkinWeights(om.MPxCommand):
    COMMAND_NAME = 'bmSmoothSkinWeights'
    undoQueue = []
    iterations = 3
    dagPath = om.MDagPath()
    selectedComponents = om.MObject()
    pinBorderVerts = False
    pruneWeightsBelow = 0.005

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
        _syntax.addFlag('i', 'iterations', om.MSyntax.kLong)
        _syntax.addFlag('pin', 'pinBorderVerts', om.MSyntax.kBoolean)
        _syntax.addFlag('pwb', 'pruneWeightsBelow', om.MSyntax.kDouble)

        return _syntax

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

        if argParser.isFlagSet('iterations'):
            self.iterations = int(argParser.flagArgumentInt('iterations', 0))
        if argParser.isFlagSet('i'):
            self.iterations = int(argParser.flagArgumentInt('i', 0))

        if argParser.isFlagSet('pinBorderVerts'):
            self.pinBorderVerts = argParser.flagArgumentBool('pinBorderVerts', 0)
        if argParser.isFlagSet('pin'):
            self.pinBorderVerts = argParser.flagArgumentBool('pin', 0)

        if argParser.isFlagSet('pruneWeightsBelow'):
            self.pruneWeightsBelow = args.flagArgumentDouble('pruneWeightsBelow', 0)
        if argParser.isFlagSet('pwb'):
            self.pruneWeightsBelow = args.flagArgumentDouble('pwb', 0)

        # get the selected components
        self.dagPath, self.selectedComponents = self.getSelectedVertexIDs()

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

        emptyComponent = om.MObject()
        # for the starting weights, we need to get weights on every vertex so we have neighbors
        originalWeights = fnSkinCluster.getWeights(self.dagPath, emptyComponent, influenceIndices)
        oldWeights = om.MDoubleArray(originalWeights)
        newWeights = oldWeights

        for iteration in xrange(self.iterations):
            # newWeights just starts as a copy of oldWeights
            newWeights = list(om.MDoubleArray(oldWeights))

            # iterate over the selected verts
            itVerts = om.MItMeshVertex(self.dagPath, self.selectedComponents)

            while not itVerts.isDone():
                isBoundaryVertex = itVerts.onBoundary()
                # option to skip evaluation on boundary vertices
                if self.pinBorderVerts and isBoundaryVertex:
                    itVerts.next()
                else:
                    neighborVerts = itVerts.getConnectedVertices()
                    neighborWeightSums = [0.0] * numInfluences
                    numNeighbors = len(neighborVerts)
                    vertId = itVerts.index()

                    for i in xrange(numNeighbors):
                        v = neighborVerts[i]
                        # get these vertex's weights and add them to our weight sums
                        for j in xrange(numInfluences):
                            neighborWeightSums[j] += oldWeights[(v * numInfluences) + j]

                    smoothedWeights = [w / float(numNeighbors) for w in neighborWeightSums]
                    weightSum = float(sum(smoothedWeights))

                    normalizedWeights = [0.0] * numInfluences
                    for smoothWeightIndex in xrange(numInfluences):
                        n = smoothedWeights[smoothWeightIndex] / weightSum
                        if n > self.pruneWeightsBelow:
                            normalizedWeights[smoothWeightIndex] = n
                        # otherwise it will stay at 0.0

                    # normalizedWeight = [w / float(weightSum) for w in smoothedWeight]

                    newWeights[(vertId * numInfluences): (vertId * numInfluences) + numInfluences] = normalizedWeights

                    itVerts.next()

            # instead of setting/getting the skin cluster weights each iteration, we just need to set this variable
            oldWeights = newWeights

        newWeightsDoubleArray = om.MDoubleArray()
        for w in newWeights:
            newWeightsDoubleArray.append(w)

        # TODO: only set weights on the passed components
        fnSkinCluster.setWeights(self.dagPath, emptyComponent, influenceIndices, newWeightsDoubleArray)

        self.undoQueue.append((skinClusterMObject, originalWeights, self.dagPath, influenceIndices))

    def undoIt(self):
        if self.undoQueue:
            fnSkinCluster = omAnim.MFnSkinCluster(self.undoQueue[-1][0])
            oldWeights = self.undoQueue[-1][1]
            meshDagPath = self.undoQueue[-1][2]
            influenceIndices = self.undoQueue[-1][3]
            # when undoing, we have to set weights on every component
            components = om.MObject()
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
