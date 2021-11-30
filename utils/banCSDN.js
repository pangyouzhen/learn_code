// ==UserScript==
// @name         banCSDN
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        https://www.baidu.com/*
// @match        *://www.baidu.com/*
// @icon         data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==
// @grant        none
// ==/UserScript==
(function () {

    // class="c-abstract"
    const containStr = "csdn已为您找到";
    const titleEndsWith = "CSDN"

    let contentDom = document.getElementById("content_left");
    if (contentDom == null) return;

    let containerDom = contentDom.getElementsByClassName("c-container");
    if (containerDom == null) return;

    for (let i = containerDom.length - 1; i >= 0; i--) {
        let titleDom = containerDom[i].getElementsByClassName("t");
        if (titleDom == null || titleDom.length === 0) continue;

        let titleText = titleDom[0].innerText;
        console.log(titleText)
        if (titleText.includes(titleEndsWith)) {
            console.log("--------------")
            containerDom[i].remove();
            continue;
        }


        let descDom = containerDom[i].getElementsByClassName("c-abstract");
        if (descDom == null || descDom.length === 0) continue;

        let descText = descDom[0].innerText;
        if (descText.indexOf(containStr) !== -1) {
            containerDom[i].remove();
            continue;
        }

        let urlText = containerDom[i].getElementsByClassName("c-showurl");
        if (urlText == null || urlText.length === 0) continue;
        let urlInnerText = urlText[0].innerText;
        if (urlInnerText.includes(titleEndsWith)) {
            containerDom[i].remove();

        }
    }
})();
// Your code here...