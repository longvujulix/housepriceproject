/* Questions to answer with EDA:
- How have salaries changed over time between different groups of people?
- How are base pay, overtime pay, and benefits allocated between different groups?
- Is there any evidence of pay discrimination based on gender in this dataset?
- How is budget allocated based on different groups and responsibilities?
*/

USE salary;

-- How have salaries changed over time between different groups of people?

SELECT DISTINCT Year 
FROM salaries;

DROP VIEW IF EXISTS salary_2011;
CREATE VIEW salary_2011 AS
(SELECT Year, JobTitle, AVG(BasePay) AS "avgbasepay", AVG(TotalPay) AS "avgtotalpay"
FROM salaries
WHERE Year = 2011
GROUP BY JobTitle
HAVING AVG(BasePay) >=100000
ORDER BY year, avgbasepay DESC, avgtotalpay DESC);

DROP VIEW IF EXISTS salary_2012;
CREATE VIEW salary_2012 AS
SELECT Year, JobTitle, AVG(BasePay) AS "avgbasepay", AVG(TotalPay) AS "avgtotalpay"
FROM salaries
WHERE Year = 2012
GROUP BY JobTitle
HAVING AVG(BasePay) >=100000
ORDER BY year, avgbasepay DESC, avgtotalpay DESC;

DROP VIEW IF EXISTS salary_2013;
CREATE VIEW salary_2013 AS
SELECT Year, JobTitle, AVG(BasePay) AS "avgbasepay", AVG(TotalPay) AS "avgtotalpay"
FROM salaries
WHERE Year = 2013
GROUP BY JobTitle
HAVING AVG(BasePay) >=100000
ORDER BY year, avgbasepay DESC, avgtotalpay DESC;

DROP VIEW IF EXISTS salary_2014;
CREATE VIEW salary_2014 AS
SELECT Year, JobTitle, AVG(BasePay) AS "avgbasepay", AVG(TotalPay) AS "avgtotalpay"
FROM salaries
WHERE Year = 2014
GROUP BY JobTitle
HAVING AVG(BasePay) >=100000
ORDER BY year, avgbasepay DESC, avgtotalpay DESC;

SELECT * 
FROM salary_2011 a
LEFT JOIN (SELECT Year, JobTitle, avgbasepay, avgtotalpay FROM salary_2012) AS b
ON a.JobTitle = b.JobTitle
LEFT JOIN (SELECT Year, JobTitle, avgbasepay, avgtotalpay FROM salary_2013) AS c
ON b.JobTitle = c.JobTitle
LEFT JOIN (SELECT Year, JobTitle, avgbasepay, avgtotalpay FROM salary_2014) AS d
ON c.JobTitle = d.JobTitle;

-- How are base pay, overtime pay, and benefits allocated between different groups?
SELECT JobTitle, SUM(BasePay) AS SumBase, SUM(OvertimePay) AS SumOvertime, SUM(Benefits) AS SumBenefit
FROM salaries
GROUP BY JobTitle
ORDER BY SumBase DESC, SumOvertime DESC, SumBenefit DESC
LIMIT 10;