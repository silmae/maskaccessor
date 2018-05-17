# MaskAccessor
Mask for xarray DataArrays. 

You can install all the required packages and VisualisorAccessor by 
`pip install -e .`

You can import it by
`import maskacc`

Every xarray.DataArray (assume its name is `cube`) that you make after importing maskacc contains a property named `M`

You can use the properties or functions of MaskAccessor by `cube.M.<desired property or function>`

Check out the example notebook for detailed example.


Citation: Annala, L., Eskelinen, M. A., Hämäläinen, J., Riihinen, A., and Pölönen, I.: PRACTICAL APPROACH FOR HYPERSPECTRAL IMAGE PROCESSING IN PYTHON, Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLII-3, 45-52, https://doi.org/10.5194/isprs-archives-XLII-3-45-2018, 2018. 


