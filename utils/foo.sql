SELECT 
    a.Audit_ID, 
    c.FC_Name, 
    COUNT(c.FC_Name) AS 'No_of_Findings'
FROM Audit AS A
LEFT OUTER JOIN Findings F 
    ON a.Audit_ID = f.AU_ID
LEFT OUTER JOIN FindingCategories C 
    ON C.FC_ID = F.Findings_category_ID
LEFT OUTER JOIN GovernmentAgencies GA 
    ON GA.GA_ID = a.GA_ID
    AND (@Agency IS NULL OR ga.GA_LegalName = @Agency)
LEFT OUTER JOIN AuditType AT 
    ON AT.AuditType_ID = a.AuditType_ID
    AND AT.Audit_Year = @Year - 1
    AND (@AuditType IS NULL OR @AuditType = AT.AuditType_Category)
GROUP BY a.Audit_Year, c.FC_Name, a.Audit_ID