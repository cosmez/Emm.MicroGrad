namespace Emm.MicroGrad;
public record Neuron
{
    public Value[] W { get; set; }
    public Value B { get; set; }
    public Neuron(int n)
    {
        var rnd = new Random();
        W = new Value[n];
        for (int i = 0; i < n; i++)
        {
            W[i] = new Value((rnd.NextSingle() * 2) - 1);
        }
        
        B = new Value((rnd.NextSingle() * 2) - 1);
    }

    public Value[] Parameters()
    {
        var result = new Value[W.Length + 1];
        result[0] = B;
        for (int i = 0; i < W.Length; i++)
        {
            result[i+1] = W[i];
        }

        return result;
    }

    public Value Apply(Value[] values)
    {
        //var sums = W.Zip(values)
        //    .Select(p => p.First * p.Second);
        var o = B;
        foreach (var zip in W.Zip(values))
        {
            o = o + (zip.First * zip.Second);
        }

        var result = o.TanH();
        return result;
    }
}
