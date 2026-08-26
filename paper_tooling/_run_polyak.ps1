$py="C:\Users\catst\AppData\Local\Python\pythoncore-3.14-64\python.exe"
Set-Location "D:\Area-and-Distribution-Free-Estimator-for-TSP"
& $py -u paper_tooling\polyak_tsplib_timing.py --kmax 100 --repeats 11
& $py -u paper_tooling\polyak_tsplib_timing.py --kmax 500 --repeats 3
Write-Output "POLYAK_DONE"
