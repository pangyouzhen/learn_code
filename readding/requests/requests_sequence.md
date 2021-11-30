```sequence
User->api:get
api->api:request
api->Session: request
Session->Session:prepare_request
Session->Session:send
Session->HTTPAdapter:send
HTTPAdapter->PoolManager:get_connnect
PoolManager-->HTTPAdapter:return
HTTPAdapter->HTTPAdapter:cert_verify
HTTPAdapter->HTTPConnectionPool:urlopen
HTTPConnectionPool-->HTTPAdapter:return
HTTPAdapter->HTTPAdapter:build_response
HTTPAdapter-->Session:return
Session-->api:return
api->User:return
```