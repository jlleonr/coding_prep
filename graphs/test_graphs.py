import unittest
import graphs


class TestGraphs(unittest.TestCase):
    '''
    Test the Depth First Traversal Algorithm using recursion
    '''
    def test_dftRecursive(self):
        self.assertEqual(['a', 'b', 'd', 'c', 'e'], 
                        graphs.depthFirstTraversalRecursive(graph=graphs.graph1, source="a"))

        self.assertListEqual(['DFW', 'SC', 'MIA', 'SJU', 'ATL', 'SJU'], 
                            graphs.depthFirstTraversalRecursive(graph=graphs.flights, source="DFW"))

    '''
    Test the Depth First Traversal algorith iteratively
    '''
    def test_dftIterative(self):
        self.assertEqual(['a', 'c', 'e', 'b', 'd'], 
                        graphs.depthFirstTraversalIterative(graph=graphs.graph1, source="a"))

    '''
    Test that there's a path from source to destination using Depth First Traversal
    '''
    def test_hasPathDFTRecursive(self):
        self.assertTrue(graphs.hasPathDFTRecursive(graph=graphs.graph1, src="a", dst="e"))
        self.assertFalse(graphs.hasPathDFTRecursive(graph=graphs.graph1, src="c", dst="f"))


if __name__ == "_main_":
    unittest.main()