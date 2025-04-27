# Distributed Spectrum challenge - 2025 Natsec hackathon at Shack 15

Make predictions on transmitters in as few steps as possible

## Background and goals

Lifted from the discord:

Hey all - welcome to the RF foxhunting challenge. A group of operators have landed in San Francisco and are trying to map out threats based on their RF signature. They only have low cost commercial RF sensing hardware, so they can’t use techniques like direction finding or time difference of arrival to localize the transmitters. What they will do instead is, starting from their landing point, walk through the city and sample the received power from the transmitter. Additionally, they have been able to prepare ahead of time - that means they have a map of SF, and any other publicly available information that’s helpful. Your challenge is to create an algorithmic agent - a “hunter” - that creates the tightest localization in the shortest number of steps.

This problem is extremely open ended. Some potential avenues to explore:
Build a reinforcement learner
Build a propagation model
Integrate publicly available data
Combine any of this, or anything else - nothing is off the table

The code checked in here works in conjunction with this repo: https://github.com/DistributedSpectrum/tx-hunt

## Ideas and execution ( short version )

First was the exploration of the training, which consisted of a series of walks with details of coordinates, signal strength as target transmitter

The first thing to notice was the prevalence of no signal data points ( rssi ~ -1000 )

Those were filtered out and the remaining data sent to a model ( RandomForest ) to make predictions about transmitter locations.  This was not successfull

There was a map included in png format that gave us the ability to figure out navigable streets

The ultimate solution was to construct a graph that could inform the agent where they could go.  Four checkpoints were set around the city and the agent walked through them and sent a prediction once they found two successive signals > -125.  The agent ceased walking after the prediction.

Pseudocode:

For each path
  - build spanning tree
  - determine path
  - At each point
    - measure rssi
    - if this is the second successive rssi > -125, make a prediction and stop
      


