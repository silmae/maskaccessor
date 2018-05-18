# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:21:08 2017

@author: Leevi Annala
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:22:47 2017

@author: Leevi Annala
"""

import unittest
import numpy as np
import xarray as xr
from maskacc import MaskAccessor
#from __functions import select_last_outer_layer_left, \
#                                    select_first_outer_layer_right, select_next, \
#                                   select_previous, select_x_values, \
#                                    select_y_values, s, mask_to_pixels
import random

class test(unittest.TestCase):

    def test__init__and_reset(self):
        #, xarray_obj, **kwargs
        arr = np.random.rand(5, 4, 3)
        cube = xr.DataArray(arr,
                            dims=['y', 'x', 'z'],
                            coords={'y': [1, 2, 3, 4, 5],
                                    'x': [1, 2, 3, 4],
                                    'z': [1, 2, 3]})
        exp_dims = ['x', 'z']
        exp_dims_dict = {'x': 1, 'z': 2}
        exp_shape = (4, 3)
        exp_no_mask_dims = ['y']
        self.assertTrue(hasattr(cube, 'M'))
        self.assertTrue(cube.M.dims == exp_dims)
        self.assertTrue(cube.M.dims_dict == exp_dims_dict)
        self.assertTrue(exp_shape == cube.M.shape)
        self.assertTrue(cube.M.no_mask_dims == exp_no_mask_dims)
        self.assertTrue((cube.M.mask == 1).all)
        cube.M.reset(dims=['y', 'x'],
                       matrix=[[1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1]])
        self.assertTrue((cube.M.mask == 1).all)
        self.assertTrue(hasattr(cube, 'M'))
        self.assertTrue(hasattr(cube.M, 'dims'))
        exp_dims = ['y', 'x']
        exp_dims_dict = {'y':0, 'x':1}
        exp_shape = (5, 4)
        exp_no_mask_dims = ['z']
        exp_mask = [[0, 0, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]]
        cube.M.reset(matrix=exp_mask)
        self.assertTrue(cube.M.dims == exp_dims)
        self.assertTrue(cube.M.dims_dict == exp_dims_dict)
        self.assertTrue(exp_shape == cube.M.shape)
        self.assertTrue(cube.M.no_mask_dims == exp_no_mask_dims)
        self.assertTrue((np.array(exp_mask) == cube.M.mask).all)
        cube.M.reset(matrix=[[0, 0, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 3, 1, 1],
                              [1, 1, 1, 1]])
        self.assertTrue((np.array(exp_mask) == cube.M.mask).all)
        arr = np.random.rand(5, 4, 3, 2)
        cube = xr.DataArray(arr,
                            dims=['y', 'x', 'z', 'a'],
                            coords={'y': [1, 2, 3, 4, 5],
                                    'x': [1, 2, 3, 4],
                                    'z': [1, 2, 3],
                                    'a': [1, 2]})
        exp_dims = ['z', 'a']
        exp_dims_dict = {'z': 2, 'a': 3}
        exp_shape = (3, 2)
        exp_no_mask_dims = ['y', 'x']
        self.assertTrue(hasattr(cube, 'M'))
        self.assertTrue(cube.M.dims == exp_dims)
        self.assertTrue(cube.M.dims_dict == exp_dims_dict)
        self.assertTrue(exp_shape == cube.M.shape)
        self.assertTrue(cube.M.no_mask_dims == exp_no_mask_dims)
        self.assertTrue((cube.M.mask == 1).all)

    def test_mask_set(self):
        arr = np.random.rand(5, 4, 3)
        cube = xr.DataArray(arr,
                            dims=['y', 'x', 'z'],
                            coords={'y': [1, 2, 3, 4, 5],
                                    'x': [1, 2, 3, 4],
                                    'z': [1, 2, 3]})
        with self.assertRaises(ValueError):
            cube.M.mask = [9, 3, 4, 5]
        with self.assertRaises(ValueError):
            cube.M.mask = []
        with self.assertRaises(ValueError):
            cube.M.mask = [[]]
        with self.assertRaises(ValueError):
            cube.M.mask = [[1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 1, 0],
                           [0, 0, 0, 0], [1, 0, 1, 1]]
        with self.assertRaises(ValueError):
            cube.M.mask = [[1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 1, 0]]
        with self.assertRaises(ValueError):
            cube.M.mask = [[1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 2, 3]]

        exp_mask = [[1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1]]
        exp_mask_np = np.array(exp_mask)
        cube.M.mask = exp_mask
        self.assertTrue((cube.M.mask == exp_mask_np).all)

    def test_mask_as_xarray(self):
        arr = np.random.rand(5, 4, 3)
        cube = xr.DataArray(arr,
                            dims=['y', 'x', 'z'],
                            coords={'y': [1, 2, 3, 4, 5],
                                    'x': [1, 2, 3, 4],
                                    'z': [1, 2, 3]})
        res = cube.M.mask_as_xarray()
        exp_shape = cube.M.shape
        self.assertTrue(isinstance(res, xr.DataArray))
        self.assertTrue(exp_shape == res.shape)
        self.assertTrue((res[:] == 1).all())

    def test_selected_ones_and_selected_zeros(self):
        arr = np.random.rand(5, 4, 3)
        cube = xr.DataArray(arr,
                            dims=['y', 'x', 'z'],
                            coords={'y': [1, 2, 3, 4, 5],
                                    'x': [1, 2, 3, 4],
                                    'z': [1, 2, 3]})
        cube.M.reset(dims=['y', 'x'],
                       matrix=[[1, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1],
                               [1, 1, 0, 1],
                               [1, 0, 1, 1]])
        self.assertFalse((cube.M.mask == 0).all())
        self.assertFalse((cube.M.mask == 1).all())
        cube.M.selected_zeros()
        self.assertTrue((cube.M.mask == 0).all())
        cube.M.reset(dims=['y', 'x'],
                       matrix=[[1, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1],
                               [1, 1, 0, 1],
                               [1, 0, 1, 1]])
        cube.M.selected_ones()
        self.assertTrue((cube.M.mask == 1).all())

    def test_mask_to_pixels(self):
        arr = np.random.rand(5, 4, 3)
        cube = xr.DataArray(arr,
                            dims=['y', 'x', 'z'],
                            coords={'y': [1, 2, 3, 4, 5],
                                    'x': [1, 2, 3, 4],
                                    'z': [1, 2, 3]})
        res = cube.M.mask_to_pixels()
        exp_res = np.array([[0, 0],
                            [0, 1],
                            [0, 2],
                            [1, 0],
                            [1, 1],
                            [1, 2],
                            [2, 0],
                            [2, 1],
                            [2, 2],
                            [3, 0],
                            [3, 1],
                            [3, 2]])
        self.assertTrue((exp_res == res).all())

        cube.M.selected_zeros()
        res = cube.M.mask_to_pixels()
        exp_res = np.array([[]])
        self.assertTrue((exp_res == res).all())

        cube.M.reset(matrix=[[0,0,0],[1,1,1],[0,0,0],[0,0,0]])
        res = cube.M.mask_to_pixels()
        exp_res = np.array([[1, 0],
                            [1, 1],
                            [1, 2]])
        self.assertTrue((exp_res == res).all())

        cube.M.reset(matrix=[[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
        res = cube.M.mask_to_pixels()
        exp_res = np.array([[0, 1],
                            [1, 1],
                            [2, 1],
                            [3, 1]])
        self.assertTrue((exp_res == res).all())

        cube.M.reset(matrix=[[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1]])
        res = cube.M.mask_to_pixels()
        exp_res = np.array([[0, 1],
                            [1, 0],
                            [1, 1],
                            [2, 1],
                            [3, 1],
                            [3, 2]])
        self.assertTrue((exp_res == res).all())

    def test_to_list(self):
        arr = np.random.rand(5, 4, 3)
        cube = xr.DataArray(arr,
                            dims=['y', 'x', 'z'],
                            coords={'y': [1, 2, 3, 4, 5],
                                    'x': [1, 2, 3, 4],
                                    'z': [1, 2, 3]})
        res = cube.M.to_list()
        exp_res = [cube.data[:, 0, 0].tolist(),
                   cube.data[:, 0, 1].tolist(),
                   cube.data[:, 0, 2].tolist(),
                   cube.data[:, 1, 0].tolist(),
                   cube.data[:, 1, 1].tolist(),
                   cube.data[:, 1, 2].tolist(),
                   cube.data[:, 2, 0].tolist(),
                   cube.data[:, 2, 1].tolist(),
                   cube.data[:, 2, 2].tolist(),
                   cube.data[:, 3, 0].tolist(),
                   cube.data[:, 3, 1].tolist(),
                   cube.data[:, 3, 2].tolist()]
        self.assertTrue(res == exp_res)

        cube.M.selected_zeros()
        res = cube.M.to_list()
        exp_res = []
        self.assertTrue(exp_res == res)

        cube.M.reset(matrix=[[0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]])
        res = cube.M.to_list()
        exp_res = [cube.data[:, 1, 0].tolist(),
                   cube.data[:, 1, 1].tolist(),
                   cube.data[:, 1, 2].tolist()]

        self.assertTrue(exp_res == res)

        cube.M.reset(matrix=[[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
        res = cube.M.to_list()
        exp_res = [cube.data[:, 0, 1].tolist(),
                   cube.data[:, 1, 1].tolist(),
                   cube.data[:, 2, 1].tolist(),
                   cube.data[:, 3, 1].tolist()]
        self.assertTrue(exp_res == res)


        cube.M.reset(matrix=[[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1]])
        res = cube.M.to_list()
        exp_res = [cube.data[:, 0, 1].tolist(),
                   cube.data[:, 1, 0].tolist(),
                   cube.data[:, 1, 1].tolist(),
                   cube.data[:, 2, 1].tolist(),
                   cube.data[:, 3, 1].tolist(),
                   cube.data[:, 3, 2].tolist()]
        self.assertTrue(exp_res == res)

    def test_where_masked(self):
        data = [[[1, 2, 2, 3], [1, 3, 2, 5]],
                [[3, 2, 2, 2], [2, 6, 4, 5]],
                [[5, 78, 32, 5], [1, 32, 4, 7]]]
        cube = xr.DataArray(data,
                            dims=['y', 'x', 'z'],
                            coords={'y': [1, 2, 3],
                                    'x': [1, 2],
                                    'z': [1, 2, 3, 4]})
        res = cube.M.where_masked()
        self.assertTrue((res == cube).all())
        self.assertTrue((res.coords['y'] == cube.coords['y']).all())
        self.assertTrue((res.coords['x'] == cube.coords['x']).all())
        self.assertTrue((res.coords['z'] == cube.coords['z']).all())

        # A test for keep_mask here with the same cube
        res = cube.M.where_masked(keep_mask=True)
        self.assertTrue((res.M.mask == cube.M.mask).all())


        cube.M.reset(matrix=[[0, 0, 0, 0], [1, 1, 1, 1]])
        res = cube.M.where_masked()
        exp_data = [[[1, 3, 2, 5]],
                    [[2, 6, 4, 5]],
                    [[1, 32, 4, 7]]]
        exp = xr.DataArray(exp_data,
                            dims=['y', 'x', 'z'],
                            coords={'y': [1, 2, 3],
                                    'x': [2],
                                    'z': [1, 2, 3, 4]})
        self.assertTrue((res.data == exp.data).all())
        self.assertTrue((res.coords['y'].data == exp.coords['y'].data).all())
        self.assertTrue((res.coords['x'].data == exp.coords['x'].data).all())
        self.assertTrue((res.coords['z'].data == exp.coords['z'].data).all())

        # A test for keep_mask here with the same cube
        res = cube.M.where_masked(keep_mask=True)
        self.assertTrue((res.M.mask == [1, 1, 1, 1]).all())

        cube.M.reset(matrix=[[1, 0, 0, 0], [1, 1, 1, 1]])
        res = cube.M.where_masked()
        exp_data = [[[1, np.nan, np.nan, np.nan], [1, 3, 2, 5]],
                    [[3, np.nan, np.nan, np.nan], [2, 6, 4, 5]],
                    [[5, np.nan, np.nan, np.nan], [1, 32, 4, 7]]]
        exp = xr.DataArray(exp_data,
                           dims=['y', 'x', 'z'],
                           coords={'y': [1, 2, 3],
                                   'x': [1, 2],
                                   'z': [1, 2, 3, 4]})
        self.assertTrue(np.allclose(res.data, exp.data, equal_nan=True))
        self.assertTrue((res.coords['y'].data == exp.coords['y'].data).all())
        self.assertTrue((res.coords['x'].data == exp.coords['x'].data).all())
        self.assertTrue((res.coords['z'].data == exp.coords['z'].data).all())

        # A test for keep_mask here with the same cube
        res = cube.M.where_masked(keep_mask=True)
        self.assertTrue((res.M.mask == [[1, 0, 0, 0], [1, 1, 1, 1]]).all())

        cube.M.reset(matrix=[[0, 1, 0, 0], [0, 1, 0, 0]])
        res = cube.M.where_masked()
        exp_data = [[[2], [3]],
                    [[2], [6]],
                    [[78], [32]]]
        exp = xr.DataArray(exp_data,
                           dims=['y', 'x', 'z'],
                           coords={'y': [1, 2, 3],
                                   'x': [1, 2],
                                   'z': [2]})

        self.assertTrue(np.allclose(res.data, exp.data, equal_nan=True))
        self.assertTrue((res.coords['y'].data == exp.coords['y'].data).all())
        self.assertTrue((res.coords['x'].data == exp.coords['x'].data).all())
        self.assertTrue((res.coords['z'].data == exp.coords['z'].data).all())

        # A test for keep_mask here with the same cube
        res = cube.M.where_masked(keep_mask=True)
        self.assertTrue((res.M.mask ==[[1], [1]]).all())

        cube.M.reset(matrix=[[0, 1, 0, 0], [1, 1, 0, 0]])
        res = cube.M.where_masked()
        exp_data = [[[np.nan, 2], [1, 3]],
                    [[np.nan, 2], [2, 6]],
                    [[np.nan, 78], [1, 32]]]
        exp = xr.DataArray(exp_data,
                           dims=['y', 'x', 'z'],
                           coords={'y': [1, 2, 3],
                                   'x': [1, 2],
                                   'z': [1, 2]})

        self.assertTrue(np.allclose(res.data, exp.data, equal_nan=True))
        self.assertTrue((res.coords['y'].data == exp.coords['y'].data).all())
        self.assertTrue((res.coords['x'].data == exp.coords['x'].data).all())
        self.assertTrue((res.coords['z'].data == exp.coords['z'].data).all())

        # A test for keep_mask here with the same cube
        res = cube.M.where_masked(keep_mask=True)
        self.assertTrue((res.M.mask == [[0, 1], [1, 1]]).all())

    def test__select_value(self):
        arr = np.random.rand(5, 4, 3)
        cube = xr.DataArray(arr,
                            dims=['y', 'x', 'z'],
                            coords={'y': [1, 2, 3, 4, 5],
                                    'x': [1, 2, 3, 4],
                                    'z': [1, 2, 3]})
        cube.M.selected_zeros()
        cube.M._select_value([[0,0]],1 )
        res = cube.M.mask
        exp = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertTrue((res == exp).all())

        cube.M._select_value([[1, 0], [2, 0]], 1)
        res = cube.M.mask
        exp = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]])
        self.assertTrue((res == exp).all())

        cube.M._select_value([[0, 0], [1, 0], [2, 0]], 0)
        res = cube.M.mask
        exp = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertTrue((res == exp).all())

        cube.M._select_value([[1, 1], [0, 2]], 1)
        res = cube.M.mask
        exp = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0]])
        self.assertTrue((res == exp).all())

        with self.assertRaises(ValueError):
            cube.M._select_value([[0, 0], [1, 1], [0, 2]], 3)

        with self.assertRaises(IndexError):
            cube.M._select_value([[0, -4], [1, 1], [0, 2]], 1)

        with self.assertRaises(IndexError):
            cube.M._select_value([[0, 3], [1, 1], [0, 2]], 1)

        with self.assertRaises(IndexError):
            cube.M._select_value([[0, 0], [-5, 1], [0, 2]], 1)

        with self.assertRaises(IndexError):
            cube.M._select_value([[0, 30], [1, 1], [5, 2]], 1)

        with self.assertRaises(TypeError):
            cube.M._select_value('Kissa', 1)

        with self.assertRaises(TypeError):
            cube.M._select_value([2, 3, 4], 1)

        with self.assertRaises(TypeError):
            cube.M._select_value([[[2, 2],[2, 2]], [[2, 2],[2, 2]]], 1)

        with self.assertRaises(TypeError):
            cube.M._select_value([[2, 2],[2, 2],[2, 2],[2, 2, 3]], 1)

        self.assertTrue((res == exp).all())


        cube.M.selected_zeros()
        cube.M._select_value([-1, -1], 1)
        res = cube.M.mask
        exp = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.assertTrue((res == exp).all())

if __name__ == '__main__':
    unittest.main()