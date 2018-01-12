# MaskAccessor
Mask for xarray DataArrays. 

You can install all the required packages and VisualisorAccessor by 
`pip install -e .`

You can import it by
`import maskacc`

Every xarray.DataArray (assume its name is `cube`) that you make after importing maskacc contains a property named `M`

You can use the properties or functions of MaskAccessor by `cube.M.<desired property or function>`

Check out the example notebook for detailed example.