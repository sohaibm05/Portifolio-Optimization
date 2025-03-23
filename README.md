# Portifolio-Optimization
Data Structures and Algorithms: Project Proposal



Team Title: Portfolio Optimization

Team Members: Syed Ibad Asim, Sohaib Muhammad, Rana Mohammad Sarib Khan

Application: Portfolio optimization is the process of selecting the best possible combination of assets to maximize returns while minimizing risk. The goal is to achieve the highest possible return for a fixed amount of risk, which may be achieved through varied means such as risk-management strategies like diversification or asset allocation adjustments, or techniques like mean-variance optimization.


Portfolio management requires constantly changing numerical values, such as stock prices, return and risk metrics, etc. As such, an algorithm capable of handling its optimization requires a data structure capable of changing and saving numerical values efficiently.

Data Structure To Be Used: Fenwick Binary Tree

Implementation: Fenwick tree shall be implemented as a simple, array-based structure. This will help in updating and querying prefix sums of an array, which is essential for handling daily returns of a portfolio. A Fenwick tree allows quick addition of daily returns over any period and updates them fast when one day's value changes. This results in an accurate total without redoing all the work from scratch. When querying the cumulative sum up to a certain day, i and -i are repeatedly subtracted from the index to quickly add up the required values, and this all happens in O(log n).

Relevance: A Fenwick tree is being utilized for the portfolio optimization algorithm, not only because it is more efficient than saving numerical data in a normal array (Which has a time complexity of O(n)), but also for its aforementioned utility in handling large amounts of constantly changing numerical data. An example would be going over cumulative returns (Returns from day 1 to day i). Additionally, due to fast access to the returns, the data can be easily fed into a Markowitz model which will be implemented to make the program run faster.

Real-World Utility: Portfolio optimization through the use of Fenwick tree has applications in several fields that lie at the intersection of finance and computer science. Algorithmic traders who deal in high-frequency trading need to obtain real-time updates on cumulative returns and portfolio weights. Retail investors require real-time updates on portfolio value and risk exposure. Large investment firms constantly rebalance their portfolios as the prices of stocks and assets fluctuate. Hence, any fintech apps or AI-assisted financial advisors would benefit immensely from an algorithm that utilizes a Fenwick tree on the backend.
