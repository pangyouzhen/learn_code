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
//自动登录尝试脚本2
(function () {

    GM.xmlHttpRequest({
        method: "POST",
        url: "http://www.example.net/login",
        data: "username=johndoe&password=xyz123",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        onload: function (response) {
            if (response.responseText.indexOf("Logged in as") > -1) {
                location.href = "http://www.example.net/dashboard";
            }
        }
    });

    function consoleTotalNum(totalNum) {
        console.log(totalNum);
    }

    var total_num = getSessionId();
    consoleTotalNum(total_num);

})();
// Your code here...