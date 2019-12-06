# GeometricGA

###### This genetic algorithm uses the DEAP package. The code is adapted from the "One max problem" example found in the DEAP repository.

Issues:

I want `Arch` to change its value after each generation loop. When `def main()` initiates `Arch` should be `Arch=np.array([...])`. If `g=1`, `Arch` should be a different, specified numpy array. And so on for each generation until `def main()` finishes. I tried to add `Arch` to the parameters of evaluation function like this: `evalComp(individual, Arch):` and `toolbox.register("evaluate", evalComp, Arch)` and then place the new definitions of `Arch` within the generation loop.

```
while min(fits) > 0 and g < 50:
    g = g + 1
    if g<1:
        Arch=np.array([0,0,0,0,1,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif g<2:
        Arch=np.array([0,0,0,2,2,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
    elif g<3:
        Arch=np.array([0,0,0,5,3,1,4,5,4,5,4,5,4,5,4,5,4,5,4,5])
    elif...
```

It printed the updated version of `Arch` after each generation however `evalComp(individual, Arch)` was still using the original definition of `Arch` to do its calculation. I suspect this is because `evalComp()` is outside of `main()`.
