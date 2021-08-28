// ==UserScript==
// @name         New Userscript
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        https://www.baidu.com/*
// @match        *://www.baidu.com/*
// @grant        GM_xmlhttpRequest
// @grant        GM_download
// @icon         data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==
// @grant        none
// ==/UserScript==
(function () {

    function getSessionId() {
        var data = JSON.stringify({
            "type": "telegram",
            "keyword": "快讯",
            "page": 333,
            "rn": 30,
            "os": "web",
            "sv": "7.2.2",
            "app": "CailianpressWeb"
        });

        var xhr = new XMLHttpRequest();
        xhr.withCredentials = true;


        xhr.onreadystatechange = function () {
            if (xhr.readyState === 4) {
                var json = JSON.parse(xhr.responseText);
                return json.data.stock.total_num;
            }
        }


        xhr.open("POST", "https://www.cls.cn/api/sw?app=CailianpressWeb&os=web&sv=7.5.5");
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.setRequestHeader("Access-Control-Allow-Origin","*")

        xhr.send(data);
    }

    function consoleTotalNum(totalNum){
        console.log(totalNum);
    }

    var total_num = getSessionId();
    consoleTotalNum(total_num);

})();
// Your code here...