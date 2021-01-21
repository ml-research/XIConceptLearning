# Remaining-Time-To-Process (RTPT)

RTPT class to rename your processes giving information on who is launching the
process, and the remaining time for it.
Created to be used with our AIML IRON table.

## Example
``` python
# Create RTPT object
rtpt = RTPT(name_initials='KK', experiment_name='ScriptName', max_iterations=100)

# Start the RTPT tracking
rtpt.start()

# Loop over all iterations
for epoch in range(100):

  # Perform a single experiment iteration
  loss = iteration()
  
  # Update the RTPT (subtitle is optional)
  rtpt.step(subtitle=f"loss={loss:2.2f}")
```

