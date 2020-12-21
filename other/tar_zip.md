| 文件格式 | 压缩常见命令                                        | 解压常见命令            | 备注                  |
| -------- | --------------------------------------------------- | ----------------------- | --------------------- |
| tar      | tar -cvf demo.tar ./demo/                           | tar -xvf demo.tar       | tar只是打包，不是压缩 |
| tar.gz   | tar -zcvf demo.tar.gz ./demo/                       | tar -zxvf demo.tar      |                       |
| tgz      |                                                     |                         |                       |
| tgz      | tar -cvf demo.tar ./demo/ \| pigz -p 8 > output.tgz | pigz -p 8 -d output.tgz | 多cpu用pigz           |
| bz2      | bzip -k  demo                                       | bunzip -k demo.bz2      |                       |
|          |                                                     |                         |                       |
|          |                                                     |                         |                       |
|          |                                                     |                         |                       |
|          |                                                     |                         |                       |

