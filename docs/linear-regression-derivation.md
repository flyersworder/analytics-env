# Linear Regression Model Derivation

## 1. Start with the Hypothesis Function
The linear regression model starts with a hypothesis function:
- y = β₀ + β₁x + ε
- Where:
  - y is the dependent variable (target)
  - x is the independent variable (feature)
  - β₀ is the y-intercept
  - β₁ is the slope
  - ε is the error term

## 2. Define the Cost Function (RSS)
The Residual Sum of Squares (RSS) measures how well the line fits the data:
- RSS = Σ(yᵢ - ŷᵢ)²
- RSS = Σ(yᵢ - (β₀ + β₁xᵢ))²
- Where:
  - yᵢ is the actual value
  - ŷᵢ is the predicted value
  - n is the number of observations

## 3. Derive Using Ordinary Least Squares (OLS)
To find the optimal parameters, take partial derivatives of RSS with respect to β₀ and β₁ and set them to zero:

### For β₀:
∂(RSS)/∂β₀ = -2Σ(yᵢ - (β₀ + β₁xᵢ)) = 0

### For β₁:
∂(RSS)/∂β₁ = -2Σ(xᵢ(yᵢ - (β₀ + β₁xᵢ))) = 0

## 4. Solve the Normal Equations
From these derivatives, we get two normal equations:
1. Σyᵢ = nβ₀ + β₁Σxᵢ
2. Σ(xᵢyᵢ) = β₀Σxᵢ + β₁Σ(xᵢ²)

## 5. Calculate the Parameters
Solving these equations gives us:
- β₁ = (n∑xᵢyᵢ - ∑xᵢ∑yᵢ)/(n∑xᵢ² - (∑xᵢ)²)
- β₀ = (∑yᵢ - β₁∑xᵢ)/n

## Alternative: Gradient Descent Method
If using gradient descent instead of OLS:

1. Initialize β₀ and β₁ with random values
2. Update parameters iteratively:
   - β₀ = β₀ - α * ∂(RSS)/∂β₀
   - β₁ = β₁ - α * ∂(RSS)/∂β₁
   - Where α is the learning rate

3. Continue until convergence (minimal change in parameters)

## Final Model
Once parameters are found, the final model is:
- ŷ = β₀ + β₁x

This can be used to make predictions for new x values.
