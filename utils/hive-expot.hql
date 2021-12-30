select a.*, b.*, c.*
from db1.a as a
         join db2.b as b on a.id = b.id
         join db3.c as c on b.idno = c.idno
where a.date >= "2021-11-21";