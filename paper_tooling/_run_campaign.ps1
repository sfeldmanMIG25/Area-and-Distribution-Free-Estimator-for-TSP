$py="C:\Users\catst\AppData\Local\Python\pythoncore-3.14-64\python.exe"
Set-Location "D:\Area-and-Distribution-Free-Estimator-for-TSP"
& $py -u paper_tooling\exact_solver_tsplib.py --solver concorde --cap 600 --repeats 3 --repeat-under 10
& $py -u paper_tooling\exact_solver_tsplib.py --solver lkh --cap 600 --repeats 3 --repeat-under 10
Write-Output "CAMPAIGN_DONE"
