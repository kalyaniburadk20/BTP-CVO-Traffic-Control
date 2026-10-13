# CVO Design Notes

## 1. Current Belief → Current Action

Let:

\[
b_t
\]

denote the current traffic belief at time \(t\).

The signal controller is represented by:

\[
\pi(\cdot)
\]

The current action is therefore:

\[
a_t^* = \pi(b_t)
\]

In words:

Current belief → Controller → Current action

For this project, the action space is:

\[
\mathcal{A} =
\{
\text{NS\_GREEN},
\text{EW\_GREEN}
\}
\]

Therefore, the controller receives the current belief and deterministically selects one of these two signal actions.

This current action \(a_t^*\) will later be compared with the hypothetical action \(a_t'\) obtained after incorporating a prospective new observation.

## 2. Hypothetical Belief → Hypothetical Action

Let:

\[
o_{t+1}
\]

represent a possible new observation.

The hypothetical updated belief is:

\[
b'_{t+1} = U(b_t, o_{t+1})
\]

where \(U\) is the belief-update function.

The hypothetical action is then:

\[
a'_t = \pi(b'_{t+1})
\]

In words:

Hypothetical observation → Belief update → Hypothetical belief → Controller → Hypothetical action

The same controller \(\pi\) is used for both the current belief and the hypothetical belief.

Only the information available to the controller changes; the controller itself does not change.

The hypothetical action \(a'_t\) will later be compared with the current action \(a_t^*\) to calculate CVO.

## 3. Control Value of Observation (CVO)

The Control Value of Observation (CVO) measures how much a prospective
new observation can affect the downstream traffic-signal control decision.

The core formulation is:

\[
CVO_t = D(a_t^*, a_t')
\]

where:

- \(a_t^*\) is the current action selected from the existing belief.
- \(a_t'\) is the hypothetical action selected after incorporating a
  prospective new observation.
- \(D\) is the action-divergence/value function.

The purpose of CVO is therefore not simply to measure traffic-state
change. It measures whether the new information can affect the
controller's action.

## 4. Current Action

The current action is obtained from the current belief:

\[
a_t^* = \pi(b_t)
\]

where:

- \(b_t\) is the current belief.
- \(\pi\) is the deterministic signal controller.

The current action belongs to:

\[
\mathcal{A} =
\{
\text{NS\_GREEN},
\text{EW\_GREEN}
\}
\]

## 5. Hypothetical Action

Let \(o_{t+1}\) represent a possible new observation.

The hypothetical updated belief is:

\[
b'_{t+1} = U(b_t,o_{t+1})
\]

where \(U\) is the belief-update function.

The hypothetical action is then:

\[
a_t' = \pi(b'_{t+1})
\]

The same controller is used for both the current and hypothetical
beliefs. Only the information available to the controller changes.

## 6. Initial Action-Divergence Formulation

For the first implementation, binary action divergence is used:

\[
D_{\text{binary}}(a_t^*,a_t') =
\begin{cases}
1 & \text{if } a_t^* \neq a_t' \\
0 & \text{if } a_t^* = a_t'
\end{cases}
\]

Therefore:

- Same action → CVO = 0
- Different action → CVO = 1

This is the initial operational implementation of CVO.

## 7. Decision Margin

The controller uses two queue-pressure quantities:

\[
P_{NS}=Q_N+Q_S
\]

\[
P_{EW}=Q_E+Q_W
\]

The decision margin is:

\[
M_t = |P_{NS}-P_{EW}|
\]

A small margin indicates that the current state is close to the
controller's decision boundary.

A large margin indicates that the controller is farther from the
decision boundary.

Decision margin is contextual information about action sensitivity;
it is not itself the CVO.

## 8. Relationship to H1

H1 investigates whether observation value depends on the current
traffic belief and the sensitivity of the control policy.

The working hypothesis is:

- Near a decision boundary, relatively small changes in traffic
  information may change the controller's action.
- Far from a decision boundary, larger changes may be required to
  change the action.

The boundary experiment is therefore used to study when CVO becomes
likely to change from 0 to 1.

This relationship is a hypothesis to be evaluated experimentally,
not assumed to be proven.

## 9. Limitation of Binary Action Divergence

Binary divergence is simple and directly measures whether the
controller's action changes.

Advantages:

- Simple to implement.
- Easy to explain.
- Easy to reproduce.
- Directly measures action change.

Limitation:

Binary divergence does not represent how close the controller was to
changing its action.

For example, two cases may both produce CVO = 0 even though one state
is very far from the decision boundary and the other is extremely
close to it.

Therefore:

- CVO represents action divergence.
- Decision margin provides context about decision-boundary proximity.

The binary formulation is an initial implementation, not the final
research formulation.

## 10. Decision on Richer Divergence

For the current implementation, binary action divergence is retained
as the initial CVO formulation.

The current signal controller is deterministic and produces one of two
discrete actions:

\[
\{\text{NS\_GREEN},\text{EW\_GREEN}\}
\]

Therefore, binary action divergence directly captures whether a
prospective observation changes the downstream control decision.

A richer divergence formulation may be investigated later if the
experimental results show that binary divergence does not capture
enough information about control sensitivity.

In particular, decision margin will be analyzed separately to provide
context about proximity to the controller decision boundary.

Thus, the current design does not claim that binary divergence is the
universally optimal CVO formulation. It is the initial validated
operational formulation for this deterministic controller.