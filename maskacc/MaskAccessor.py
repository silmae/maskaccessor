# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:14:58 2017

@author: Leevi Annala
"""

import xarray as xr
import numpy as np
import warnings
#from spectral_selector.point_chooser import point_chooser


def read_rasterio(filepath):
    '''
    Reads a raster file with rasterio to xarray
    and gives a mask to it. Probably unnesessary.
    Definitely unnesessary, probably harmful
    :param filepath: Filepath of the file
    :return: xarray DataArray
    '''
    cube = xr.open_rasterio(filepath)
    MaskAccessor(cube)
    return cube

# @xr.register_dataset_accessor('M')
@xr.register_dataarray_accessor('M')
class MaskAccessor(object):
    '''
    This is MaskAccessor to be pasted to xarray.dataarray.
    If cube is dataarray, the MaskAccessor is to be used the following way:
    MaskAccessor(cube) # Puts Accessor to cube
    cube.mask.<function_name_here> # uses some mask function
    
    The cube should have coordinates and dims properly attached.
    TODO: FInd out if this works as dataset_accessor
    '''
    def __init__(self, xarray_obj, **kwargs):
        '''
        Initializes the MaskAccessor, with xarray.DataArray and a matrix
        TODO: test
        :param xarray_obj: carray Dataarray
        :param kwargs:  dims: The dimensions you want to use for a mask
                        matrix: the mask as a matrix
        '''
        # right_type = minimum requirements for this class to work
        # No guarantee is given for right output, you should stick to
        # xarray DataArray
        right_type = hasattr(xarray_obj, 'dims') and \
                     hasattr(xarray_obj, 'shape') and \
                     hasattr(xarray_obj, 'isel') and \
                     hasattr(xarray_obj, 'where')
        if not right_type:
            raise TypeError('Cube should be of type xarray.DataArray, ' +
                            'or at least have attributes dims, shape, isel ' +
                            'and where.')
        self._obj = xarray_obj
        self.refresh(**kwargs)
        
    def refresh(self, **kwargs):
        '''
        Function that refreshesh the MaskAccessor with a matrix and masked
        dimensions.
        TODO: TEST
        :param kwargs:  dims: The dimensions you want to use for a mask
                        matrix: the mask as a matrix
        :return: Nothing
        '''
        # If dims are given we use them, else we use the two last from the
        # objects dims. From the dimensions we find out the shape of the mask.

        def __make_attributes():
            '''
            Makes dims_dict, shape, no_mask_dims and no_mask_dims_dict
            attributes
            :return: nothing
            '''
            first_dim = self._dims[0]
            second_dim = self.dims[1]
            obj_dims = list(self._obj.dims)
            first_dim_index = obj_dims.index(first_dim)
            second_dim_index = obj_dims.index(second_dim)

            # dims_dict
            self._dims_dict = {
                first_dim: first_dim_index,
                second_dim: second_dim_index
            }

            # shape
            self._shape = (
                self._obj.shape[first_dim_index],
                self._obj.shape[second_dim_index]
            )

            # no_mask_dims
            self._no_mask_dims = []
            for i, dim in enumerate(obj_dims):
                if not (i == first_dim_index or i == second_dim_index):
                    self._no_mask_dims.append(dim)

            # no_mask_dims_dict
            self._no_mask_dims_dict = {}
            for dim in self._no_mask_dims:
                self._no_mask_dims_dict[dim] = obj_dims.index(dim)

        # If dims are given we use them, else we use the two last dimensions
        if 'dims' in kwargs.keys():
            # print('if')
            dims_self = []
            dims = kwargs['dims']
            if not set(dims) < set(self._obj.dims):
                raise ValueError('Some of your dims is not in xarray ' +
                                 'dimensions.')
            # We make sure that the dims are in desired order, not necessarily
            # the order user give us
            for dim in self._obj.dims:
                if dim in dims:
                    # print(dim)
                    dims_self.append(dim)

            if not (len(dims) == 2 and len(dims_self) == 2):
                raise ValueError('MaskAccessor only supports ' +
                                 'two dimensional masks.')
            # print(dims_self)
            self._dims = dims_self
            __make_attributes()
        elif hasattr(self, 'dims'):
            pass
        else:
            # print('else')
            self._dims = [self._obj.dims[-2], self._obj.dims[-1]]
            __make_attributes()
        try:
            original_mask = self._mask
            if original_mask.shape != self._shape:
                original_mask = None
        except AttributeError:
            original_mask = None
        changed = False
        # If matrix is given and is of right shape we use it, else we use
        # Ones mask.
        if 'matrix' in kwargs.keys():
            mask_candidate = kwargs['matrix']
            mask_candidate = np.array(mask_candidate)
            # print(self.shape)
            # print(matrix.shape)
            if mask_candidate.shape == self._shape and \
                    ((mask_candidate == 1) + (mask_candidate == 0)).all():
                # if matrix is the right shape we put it to self_mask, else
                # we destroy it
                self._mask = mask_candidate
                changed = True
            elif original_mask is not None:
                warnings.warn('Given matrix was not the right shape, or it' +
                              ' contained something else than one or zero, ' +
                              'doing nothing instead, since there was ' +
                              'original mask with right shape. ')
            else:
                warnings.warn('Given matrix was not the right shape, or it ' +
                              'contained something else than one or zero, ' +
                              'using ones instead.')
                
        if original_mask is None and not changed:
            # Here we put the fallback value to self._mask
            self._mask = np.ones(self._shape)


    @property
    def no_mask_dims(self):
        '''
        Gets the property no_mask_dims
        TODO: test
        :return: mask property no_mask_dims
        '''
        return self._no_mask_dims
    
    @no_mask_dims.setter
    def no_mask_dims(self, *args):
        '''
        Raises an AssertionError if someone tries to set no_mask_dims.
        They are just for getting, no setting pls.
        TODO: test
        :return: nothing
        '''
        raise AssertionError('You cannot change the dims without changing ' +
                             'mask as well. Try refresh -method instead.')
    
        
    @property
    def dims(self):
        '''
        Gets the property dims
        TODO: test
        :return: mask property dims
        '''
        return self._dims
    
    @dims.setter
    def dims(self, *args):
        '''
        Raises an AssertionError if someone tries to set dims.
        They are just for getting, no setting pls.
        TODO: test
        :return: nothing
        '''
        raise AssertionError('You cannot change the dims without changing ' +
                             'mask as well. Try refresh -method instead.')
    
            
    @property
    def shape(self):
        '''
        Gets the property shape
        TODO: test
        :return: mask property shape
        '''
        return self._shape
    
    @shape.setter
    def shape(self, *args):
        '''
        Raises an AssertionError if someone tries to set shape.
        They are just for getting, no setting pls.
        TODO: test
        '''
        raise AssertionError('You cannot change the shape without changing ' +
                             'mask as well. Try refresh -method instead.')
    
    @property
    def dims_dict(self):
        '''
        Gets the property dims_dict
        TODO: test
        :return: mask property dims_dict
        '''
        return self._dims_dict
    
    @dims_dict.setter
    def dims_dict(self, *args):
        '''
        Raises an AssertionError if someone tries to set dims_dict.
        They are just for getting, no setting pls.
        TODO: test
        '''
        raise AssertionError('You cannot change the dims_dict without ' +
                             'changing mask as well. Try refresh -method ' +
                             'instead.')
    
    @property
    def mask(self):
        '''
        TODO: test
        '''
        return self._mask
        
    @mask.setter
    def mask(self, matrix):
        '''
        Setter for mask. If the suggested mask is of wrong shape, raises
        ValueError
        TODO: test
        :param matrix: The suggested mask
        :return: nothing
        '''
        matrix = np.array(matrix)
        # print(((matrix == 0) + (matrix == 1)).all())
        if matrix.shape == self._shape and ((matrix == 0) + (matrix == 1)).all():
            # if matrix is the right shape we put it to self_mask, else
            # we raise error
            self._mask = matrix
        else:
            raise ValueError('The mask should be of same size as' +
                             ' the layers of your cube and contain only ' +
                             'ones and zeros')
            
    def mask_as_xarray(self):
        '''
        Converts the mask to xarray DataArray
        :return: mask as xr.DataArray
        '''
        '''
        TODO: test
        '''
        opts = {}
        for dim in self._no_mask_dims:
            opts[dim] = 0
        ret = self._obj.isel(**opts).copy()
        ret.data = self.mask
        return ret
    
    def selected_ones(self):
        '''
        Makes the selected array contain just ones
        TODO: test
        :return: Nothing
        '''
        self._mask[:] = 1

    def selected_zeros(self):
        '''
        Makes the selected array contain just zeros
        TODO: test
        :return: Nothing
        '''
        self._mask[:] = 0

    #def select_all_value(self, value):
    #    '''
    #    Sets all pixels to spesified value
    #    TODO: test
    ##    :param value: what value is wanted to be in mask
    #    :return: nothing
    #    '''
    #    self._mask[:] = value

    def mask_to_pixels(self):
        '''
        Returns the selected array -coordinates as a list
        TODO: test
        :return: numpy array of coordinates of places in mask with one in it
        '''
        ret = []
        for i in range(len(self._mask)):
            for j in range(len(self._mask[0])):
                if self._mask[i, j] == 1:
                    first_coord = i
                    second_coord = j
                    ret.append((first_coord, second_coord))
        return np.array(ret)

    #def mask_to_spectra(self):
    #    '''
    #    Takes n*m matrix, that has binary values, and n*m*l cube, and selects
    #   from cube the places where matrix has value 1.
    #    TODO: Test
    #   :return: list of data from xarray object corresponding to each mask
    #    pixel
    #    '''
    #    ret = []
    #    for first_coord in range(len(self._mask)):
    #        for second_coord in range(len(self._mask[0])):
    #            if self._mask[first_coord, second_coord] == 1:
    #                opts = {
    #                        self._dims[0]:first_coord,
    #                        self._dims[1]:second_coord
    #                        }
    #                ret.append(self._obj.isel(**opts).data)
    #
    #    return np.array(ret)
   
    def where_masked(self):
        '''
        Drops the parts of xarray objects where mask = 0
        :return: new xarray objects without the mask = 0 -pixels
        TODO: Test
        '''
        return self._obj.where(self.mask_as_xarray(), drop=True)
        # For drop = True 
        # we need _mask to be DataArray or Dataset
        
    
    def to_list(self):
        '''
        Return selected spectra as a list.
        TODO: Test
        :return:  Returns masked data as a list
        '''
        ret = self.where_masked()
        combined_name = ''
        for i in self._obj.dims:
            combined_name = combined_name + i
        opts = {combined_name:(self._dims[0], self._dims[1])}
        ret = ret.stack(**opts)
        ret = ret.T
        ret = ret.where(ret == ret, drop=True)
        return ret.data#.tolist()

    def _select_value(self, selection, value):
        '''
        Puts given value to selected array on places defined by selection
        TODO: test
        :param selection: Selection should be a object of length 2 or a
                          length having object of objects of length 2.
        :param value:   Value you want to put in mask. Preferably 1 or 0, but
                        it's your mask.
        :return: nothing
        '''
        if not (value == 0 or value == 1):
            raise ValueError('value should be 1 or 0')
        if not len(np.shape(selection)) <= 2:
            raise TypeError('Selection should be a tuple of length ' +
                            '2 or a list of objects of length 2.')
        selection_has_length = hasattr(selection, '__len__')
        selection0_has_length = len(np.shape(selection)) > 1
        backup = self.mask.copy()
        # selection is empty list or array
        if selection_has_length:
            if len(selection) == 0:
                return

        # selection is a list of list or two dimensional array
        if selection_has_length and selection0_has_length:
            for sel in selection:
                if not len(sel) == 2:
                    self.mask = backup
                    raise TypeError('Selection should be a tuple of length ' +
                                    '2 or a list of objects of length 2.')
                second_coord = sel[1]
                first_coord = sel[0]
                try:
                    self._mask[first_coord, second_coord] = value
                except IndexError as e:
                    self._mask = backup
                    raise IndexError(e)
            return
        # selection is a list or one dimensional array
        elif selection_has_length:
            if not len(selection) == 2:
                raise TypeError('Selection should be a tuple of length ' +
                                '2 or a list of objects of length 2.' + str(selection))
            second_coord = selection[1]
            first_coord = selection[0]
            self._mask[first_coord, second_coord] = value
            return

        # Selection is of wrong type
        raise TypeError('Selection should be a tuple of length ' +
                        '2 or a list of objects of length 2.')

    def unselect(self, selection):
        '''
        Puts zeros to selected-array, to pixels specified by selection
        TODO: test
        :param selection: Places to put zeros.
        :return: nothing
        '''
        self._select_value(selection, 0)

    def select(self, selection):
        '''
        Puts ones to selected-array, to pixels specified by selection
        TODO: test
        :param selection: Places to pu ones
        :return: nothing
        '''
        self._select_value(selection, 1)
