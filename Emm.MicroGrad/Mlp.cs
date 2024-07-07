namespace Emm.MicroGrad;
/// <summary>
/// Multilayer perceptron
/// </summary>
public class Mlp
{
    public readonly Layer[] Layers;
    public Mlp(int inputs, int[] sizes)
    {
        var sz = new int[sizes.Length + 1];
        sz[0] = inputs;
        sizes.CopyTo(sz, 1);
        
        Layers = new Layer[sizes.Length];
        for (int i = 0; i < sizes.Length; i++)
        {
            Layers[i] = new Layer(sz[i], sz[i + 1]);
        }
    }

    public Value[] Parameters()
    {
        return Layers.SelectMany(l => l.Parameters()).ToArray();
    }

    public Value[] Apply(Value[] values)
    {
        foreach (var layer in Layers)
        {
            values = layer.Apply(values);
        }

        return values;
    }
}
