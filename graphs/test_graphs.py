import unittest
import graphs


class TestGraphs(unittest.TestCase):
    '''
    Test the Depth First Traversal Algorithm using recursion
    '''
    # def test_dft_recursive(self):
    #     self.assertEqual(['a', 'b', 'd', 'c', 'e'],
    #                      graphs.depth_first_traversal_recursive(
    #                          graph=graphs.graph1,
    #                          source="a"))

    #     self.assertListEqual(['DFW', 'SC', 'MIA', 'SJU', 'ATL', 'SJU'],
    #                          graphs.depth_first_traversal_recursive(
    #                              graph=graphs.flights,
    #                              source="DFW"))

    def test_dft_iterative(self):
        '''
        Test the Depth First Traversal algorith iteratively
        '''
        self.assertEqual(['a', 'c', 'e', 'b', 'd'],
                         graphs.depth_first_traversal_iterative(
                             graph=graphs.graph1,
                             source="a"))

    def test_has_path_dft_recursive(self):
        '''
        Test that there's a path from source to destination using
        Depth First Traversal
        '''
        self.assertTrue(graphs.has_path_dft_recursive(
            graph=graphs.graph1,
            src="a",
            dst="e"))
        self.assertFalse(graphs.has_path_dft_recursive(
            graph=graphs.graph1,
            src="c",
            dst="f"))


if __name__ == "_main_":
    unittest.main()
