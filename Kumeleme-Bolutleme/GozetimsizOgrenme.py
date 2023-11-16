# Fırat Üniversitesi Veri Çekme

"""
import requests
from bs4 import BeautifulSoup
import pandas as pd



url = "https://www.bkmkitap.com/kitap/cok-satan-kitaplar"
response = requests.get(url)
html_icerigi = response.content
soup = BeautifulSoup(html_icerigi,"html.parser")
fiyat = soup.find_all("div",{"class":"col col-12 currentPrice"})
isim =soup.find_all("a",{"class":"fl col-12 text-description detailLink"})
yazar = soup.find_all("a",{"class":"fl col-12 text-title"})
yayın = soup.find_all("a",{"class":"col col-12 text-title mt"})



liste = list()

for i in range(len(isim)):
    isim[i] = (isim[i].text).strip("\n").strip()
    yazar[i] = (yazar[i].text).strip("\n").strip()
    yayın[i] = (yayın[i].text).strip("\n").strip()
    fiyat[i] = (fiyat[i].text).strip("\n").replace("\nTL"," TL").strip()
    liste.append([isim[i],yazar[i],yayın[i],fiyat[i]])

df = pd.DataFrame(liste,columns = ["Kitap İsmi","Yazar","Yayın Evi","Fiyat"])
print(df)


import requests
from bs4 import BeautifulSoup
import pandas as pd 
url = "https://yazilimtf.firat.edu.tr/academic-staffs"
response = requests.get(url)
html_content = response.content
# requests.get('https://google.com', verify='/path/to/certfile')
requests.get('https://google.com', verify=False)

soup = BeautifulSoup(html_content, "html.parser")

# Akademik personel bilgilerini çekme
akademik_personel = []
for personel_div in soup.find_all("div", {"class": "column is-8"}):
    unvan = personel_div.find("h2").text.strip()
    bilgi_div = personel_div.find("p")
    ad_soyad = bilgi_div.find("strong").text.strip()
    e_posta = bilgi_div.find("a", {"class": "has-text-link"}).text.strip()
    calisma_alanlari = [alan.text.strip() for alan in bilgi_div.find_all("li")]
    
    akademik_personel.append({"Unvan": unvan, "Ad_Soyad": ad_soyad, "E_posta": e_posta, "Calisma_Alanlari": calisma_alanlari})

# DataFrame oluşturma
df = pd.DataFrame(akademik_personel)

# DataFrame'i ekrana basma
print(df)

"""


import requests 
from bs4 import BeautifulSoup
import pandas as pd 

url = "https://yazilimtf.firat.edu.tr/academic-staffs"
response = requests.get(url, verify=False)
html_icerik = response.content
soup = BeautifulSoup(html_icerik, "html.parser")

isim = soup.find_all("div", attrs= {"personnel-card-info-name"})
eposta = soup.find_all("div",attrs = "personnel-card-info-contact-mail")
calısma_alanları = soup.find_all("div", class_ = "personnel-card-info-work-places")



isimler = [isim.get_text(strip=True) for isim in soup.find_all("div", class_="personnel-card-info-name")]
epostalar = [eposta.get_text(strip=True) for eposta in soup.find_all("div", class_="personnel-card-info-contact-mail")]
calisma_alanlari = [alan.get_text(strip=True) for alan in soup.find_all("div", class_="personnel-card-info-work-places")]        


# personel_bilgisi = soup.find('div', class_='personnel-card-info-name')
# unvan = soup.find_all("h6", recursive=False).text.strip()    
# isim = soup.find_all("h6", recursive=False)[1].text.strip()
    

# df = pd.DataFrame(list(zip(isim, eposta, calisma_alanları)), 
#                   columns=["İsim","E-Posta", "Çalışma Alanları"])

liste = list()
for i in range(len(isim)):
    isim[i] = (isim[i].text).strip("\n").strip()
    eposta[i] = (eposta[i].text).strip("\n").strip()
    calısma_alanları[i] = (calısma_alanları[i].text).strip("\n").strip()
    liste.append([isim[i],eposta[i],calısma_alanları[i]])
    
    cıktı = pd.DataFrame(liste,columns=["ad","eposta","calısmalanaları"])
# Veriyi gösterme
print(cıktı)

# =============================================================================
# 
# =============================================================================

import requests 
from bs4 import BeautifulSoup
import pandas as pd 

url = "https://yazilimtf.firat.edu.tr/academic-staffs"
response = requests.get(url, verify=False)
html_icerik = response.content
soup = BeautifulSoup(html_icerik, "html.parser")

# Her bir personel etiketini bulma
personel_etiketleri = soup.find_all("div", class_="personnel-card-info-name")
liste1 = list()
# Her bir personel için işlem yapma
for personel_etiket in personel_etiketleri:
    # Ünvanı ve ismi çekme
    unvan = personel_etiket.find("h6", recursive=False).text.strip()
    isim = personel_etiket.find_all("h6", recursive=False)[1].text.strip()
    
    # E-posta ve çalışma alanlarını çekme
    eposta_etiket = personel_etiket.find_all("div", class_="personnel-card-info-contact-mail")
    eposta = eposta_etiket.get_text(strip=True) if eposta_etiket else ""
        
    calisma_alanlari_etiket = personel_etiket.find_all("div", class_="personnel-card-info-work-places")
    calisma_alanlari = calisma_alanlari_etiket.get_text(strip=True) if calisma_alanlari_etiket else ""
    liste1.append([unvan,isim,eposta,calisma_alanlari])
    print("Ünvan:", unvan)
    print("İsim:", isim)
    print("E-Posta:", eposta)
    print("Çalışma Alanları:", calisma_alanlari)
    print("\n")
    cıktı1 = pd.DataFrame(liste1,columns=["unvan","İsim","E-Posta", "Çalışma Alanları"])

# DataFrame oluşturma
df = pd.DataFrame(list(zip(unvan,isimler, epostalar, calisma_alanlari)), 
                   columns=["unvan","İsim","E-Posta", "Çalışma Alanları"])

# Veriyi gösterme
print(df)

