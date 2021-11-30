import time

from selenium import webdriver

driver = webdriver.Chrome("")

driver.get("https://www.baidu.com/")
username = driver.find_element_by_class_name("login")
username.send_keys("pangyouzhen")
username.find_element_by_class_name("")
time.sleep(5)
driver.close()
