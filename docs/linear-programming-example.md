# Linear Programming Problem: Production Planning Example

## Problem Statement
A furniture company produces tables and chairs. The company wants to maximize its profit while working within its resource constraints.

Given:
- Each table requires 4 hours of carpentry and 2 hours of finishing
- Each chair requires 3 hours of carpentry and 1 hour of finishing
- The company has 25 hours of carpentry time and 12 hours of finishing time available
- Each table earns $180 profit
- Each chair earns $120 profit
- The company must produce at least 2 tables due to an existing order
- Storage space limits production to no more than 8 pieces of furniture total

Find: How many tables and chairs should be produced to maximize profit?

## Step 1: Define Variables
Let:
- x = number of tables to produce
- y = number of chairs to produce

## Step 2: Write Objective Function
Maximize Profit = 180x + 120y

## Step 3: Identify Constraints

### Resource Constraints:
1. Carpentry time: 4x + 3y ≤ 25 (hours)
2. Finishing time: 2x + y ≤ 12 (hours)

### Other Constraints:
3. Minimum tables: x ≥ 2 (existing order)
4. Total furniture: x + y ≤ 8 (storage limit)
5. Non-negativity: x ≥ 0, y ≥ 0

## Step 4: Complete LP Model

Maximize Z = 180x + 120y

Subject to:
1. 4x + 3y ≤ 25
2. 2x + y ≤ 12
3. x ≥ 2
4. x + y ≤ 8
5. x ≥ 0
6. y ≥ 0

## Solution Approach
1. Plot constraints on a coordinate system
2. Find feasible region (area satisfying all constraints)
3. Find corner points of feasible region
4. Evaluate objective function at each corner point
5. Select point giving maximum value

## Key Points for Interview Discussion
1. **Identifying Variables**: Explain why x and y were chosen
2. **Formulating Objective Function**: Discuss why profit maximization was chosen
3. **Constraint Types**:
   - Resource constraints (carpentry and finishing time)
   - Demand constraints (minimum order)
   - Capacity constraints (storage limit)
   - Non-negativity constraints
4. **Solution Methods**:
   - Graphical method (for 2D problems)
   - Simplex method (for larger problems)
5. **Sensitivity Analysis**: How solution changes with:
   - Different profit margins
   - Different resource availability
   - Different minimum order requirements

## Common Interview Follow-up Questions
1. How would you modify the model if:
   - There was a minimum order for chairs?
   - Storage cost per unit needed to be considered?
   - Different quality grades of wood were available?
2. What assumptions does this model make?
3. How would you handle uncertainty in the profit margins?
4. What if demand was uncertain?

## Real-world Considerations
1. **Model Limitations**:
   - Assumes constant profit per unit
   - Assumes resources are fully divisible
   - Assumes perfect information
2. **Additional Factors**:
   - Setup costs
   - Storage costs
   - Labor availability
   - Market demand fluctuations
