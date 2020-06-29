# Stock-Market-Prediction
OBJECTIVE :

The aim of this project is to predict and compare the future value of the financial stocks of big companies- Apple, GE, Google, IBM, and Microsoft so that the user can easily invest in those companies stocks’ which are going to make profits in the upcoming future. 
For this machine learning is used to make prediction based on the data of current stock market indices by training on their previous data values. 
With the help of this project, the use of irregular sources for collection of information and prediction of stock market will decrease which in turn will also save the user’s valuable time.

LANGUAGE : Python

TOOLS USED: Jupyter (Anaconda Navigator)

LIBRARIES & PACKAGES USED : Pandas, Matplotlib, NumPy, Scikit-learn

METHODOLOGY :

We used machine learning which is in the recent trend and made predictions based on the data of the current stock market indices by training on their previous data values.
We collected the last 7 years of input data(date, open, high, low, close, adj close) and extract Apple, GE, Google, IBM, and Microsoft Stocks Price from Yahoo Finance for the prediction.
After that analyzed stocks using rolling mean and return rate. To get a better understanding of this, plotted it out with the help of matplotlib. In the graph, the upturns will show a favorable time to sell the stocks while the downturn shows a favorable time to buy the stocks.
Now we analyzed how one company performs relative to its competitor and return closing prices among the stock prices from Yahoo Finance.
Then we analyzed how one company's stock behaves relative to other company's stock. After that, we analyzed the competition by correlation function and percentage change. Plot ScatterPlot to view the stock return distributions between any 2 companies, say, Apple and GE.
We further improve our analysis by plotting scatter_matrix to visualize possible correlations among competing stocks.
And to prove the positive correlations, we used heat maps to visualize the correlation ranges among the competing stocks where lighter color shows that two stocks are more correlated.
Apart from correlation, we also analyze each stock’s risks and returns. Customers can buy those stocks which are at a lower level, and sell those are at the upper level, to make profit.
After that we calculate High Low Percentage and Percentage Change to predict our stocks.
Then we pre-processed our data, and start analysis using- Simple Linear Analysis, Quadratic Discriminant Analysis and K Nearest Neighbors.
At last, we plot our prediction. The successful prediction of the stock will be a great asset for the market companies and will provide real-life solutions to the problem that stock investors face in today’s time.
