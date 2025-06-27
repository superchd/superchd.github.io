# Bayes Theorem

* Representative Example
  $$
  P(H|E) = \frac{P(E|H) \cdot P(H)}{P(H) \cdot P(E|H) + P(-H) \cdot P(E|-H)  }
  $$

  > ### Definition and Points of confusion  
  >
  > * It expresses "The probability of truly having the disease among those who have received a positive diagnosis."
  >   But, I found initially it confusing because I can't understand the relationship with P(E|H) and P(H). And I don't know why these terms were used. 
  >
  >
  > ### Meaning
  >
  > * It is the ability to determine the posterior probability based on prior beliefs and also can estimate future states based on the current state.

  

  

# Bayes localization filter / Markov localization

* Definition and why we use it?

  > * We can't record all data bcs its too large. We have to reduce data or find a way to calculate very fast. So, we use Bayes theorem to estimate current state.
  > * We will only use new observation data and not record previous data. We will just leave current state not past state.
  >
  >
  > $$
  > bel(x_t) = {P(x_t|z_t, z_{1:t-1},u_{1:t},m)}
  > $$
  >
  > * z is about current information and previous information. 
  >
  >   ### Quiz
  >
  >   * how can you express this using bayes theorem?
  >     $$
  >     {P(x_t|z_t, z_{1:t-1},u_{1:t},m)}
  >     $$
  >
  >   * 
  >
  >   ### Answer
  >
  >   * $$
  >     {P(x_t|z_t, z_{1:t-1},u_{1:t},m)} = \frac {P(z_t|x_t, z_{1:t-1},u_{1:t},m) \cdot P(x_t|z_{1:t-1},u_{1:t},m)} {P(z_t|z_{1:t-1},u_{1:t},m)}
  >     $$
  >
  >   * First term of the numerator maybe just changed left-hand side term's elements. 
  >
  >   * Then, why second term of the numerator is expressed in this way?
  >
  >   * It removes Zt term!!!
  >
  >   * I have to understand how this equation is derived!
  >
  >   * The x_t which is in first term of numerator referred to the moment that robot is in x_t.
  >
  >   * So, the second term of the numerator simply means  "Robot is in x_t"
  >
  >   * The "z_t" in the denominator represents the probability that the robot will be predicted at position "z_t."
  >
  >   * If you remove posterior, this equation shows the general bayes formula.
  >
  >   * The first term of numerator means the **obeservation model** , the probablity distribution of observation vector. And, state x_t is previous observation(state).
  >
  >   * The second term of numerator means probablity distribution of x_t that no current observations are included in the motion model.
  >
  >   * $$
  >     {P(x_t|z_t, z_{1:t-1},u_{1:t},m)} = \eta \cdot{P(z_t|x_t, z_{1:t-1},u_{1:t},m) \cdot P(x_t|z_{1:t-1},u_{1:t},m)}
  >     $$
  >
  >   * $$
  >     \eta \space is \space normalized \space term 
  >     $$
  >
  >   * And etha is the product of the observation and the motion model over all possible states.





# Law of total probability

* $$
  {P(x_t|z_{1:t-1},u_{1:t},m)} = \int {{P(x_t|x_{t-1}, \space z_{1:t -1}, \space u_{1:t}, \space m)} \cdot P(x_t|z_{1:t-1}, \space u_{1:t}, \space m)} \space dx_{t-1}
  $$

* With all possible states of the previous step and we can predict where the car would be in the next step.



# Markov Assumption

* With markov assumption we can remove not essential terms.

* $$
  {P(x_t|z_{1:t-1},u_{1:t},m)} = \int {{P(x_t|x_{t-1}, \space z_{1:t -1}, \space u_{1:t}, \space m)} \cdot P(x_t|z_{1:t-1}, \space u_{1:t}, \space m)} \space dx_{t-1}
  $$

* $$
  {P(x_t|z_{1:t-1},u_{1:t},m)} = \int {{P(x_t|x_{t-1}, \space u_t, \space m)} \cdot P(x_{t-1}|z_{1:t-1}, \space u_{1:t-1}, \space m)} \space dx_{t-1}
  $$

* $$
  {P(x_t|z_{1:t-1},u_{1:t},m)} = \int {{P(x_t|x_{t-1}, \space u_t, \space m)} \cdot bel(x_{t-1}}) \space dx_{t-1}
  $$

* ### Recursive !!





# Observation Model

* Meaning : we not only detect one object, but also several objects. In that case, we use observation model. We can express these to this equation.

* Multiple observation model for each time step possible.

* $$
  {P(z_t|x_t, m)} = {P(z_t^1, ....,z_t^k|x_t, m)}
  $$

* $$
  {P(z_t^1, ....,z_t^k|x_t, m)} = \prod_{k=1}^K p(z_t^k|x_t,m)
  $$
