# Sign-Language
## 主題
手語辨識
## 套件
- Django
- Django-crispy-forms
- mediapipe
- opencv-python
- mysqlclient
- pillow
- joblib
- scikit-learn
- pipenv (optional)
## Setup
### 建立虛擬環境 (optional)
`pipenv --python 3.9.13` (python 版本依照電腦安裝之版本選擇即可)。
如果是使用 pipenv 設立虛擬環境的話，後續在執行 python 時須加上 `pipenv run`，如：`pipenv run py {filename}.py`，若是要安裝套件，則需使用 `pipenv install {package_name}`。
其餘虛擬環境請參考各自開發者文件。
### 安裝套件
安裝上述套件，惟本次專題使用 xampp 作為資料庫媒介，故需確認電腦是否已安裝 xampp 並確實將專案資料夾放入 htdocs 中
如使用 pipenv，則在 terminal 輸入 `pipenv sync` 即可。
### 啟動虛擬環境 (optional)
輸入指令 `pipenv shell` 即可啟動虛擬環境。
### 啟動專案
cd 至 `myWeb` 資料夾，並在 terminal 輸入 `pipenv py run manage.py runserver`，當 terminal 出現 **Starting development server at http://127.0.0.1:8000/** 即代表成功啟動 Django。
## Reference
https://github.com/HaoyuWang00/Mediapipe-Django-API

