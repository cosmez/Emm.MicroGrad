namespace Emm.MicroGrad;
public record Value
{
    public float Data { get; set; }
    public Value[] Previous { get; }
    public string Label { get; set; }
    public string Operation { get; }
    public float Grad { get; set; }

    public Func<float>? _backward = null;

    public Value(float data)
    {
        Data = data;
        Previous = [];
        Label = "";
        Operation = "";
    }

    public Value(float data, string label)
    {
        Data = data;
        Previous = [];
        Label = label;
        Operation = "";
    }

    public Value(float data, Value[] childreen, string operation, string label = "")
    {
        Data = data;
        Previous = childreen;
        Operation = operation;
        Label = label;
    }

    public static Value operator +(Value left, float right) => left + (new Value(right));
    public static Value operator +(float left, Value right) => (new Value(left)) + right;
    public static Value operator +(Value left, Value right)
    {
        float sum = left.Data + right.Data;
        var o = new Value(sum, [left, right], "+");
        o._backward = () =>
        {
            left.Grad += 1.0f * o.Grad;
            right.Grad += 1.0f * o.Grad;
            return 0.0f;
        };
        return o;
    }

    public static Value operator -(Value left, float right) => left - (new Value(right));
    public static Value operator -(float left, Value right) => (new Value(left)) - right;
    public static Value operator -(Value left, Value right)
    {
        return left + right.Neg();
    }

    public static Value operator *(Value left, float right) => left * (new Value(right));
    public static Value operator *(float left, Value right) => (new Value(left)) * right;
    public static Value operator *(Value left, Value right)
    {
        float sum = left.Data * right.Data;
        var o = new Value(sum, [left, right], "*");
        o._backward = () =>
        {
            left.Grad += right.Data * o.Grad;
            right.Grad += left.Data * o.Grad;
            return 0.0f;
        };
        return o;
    }

    public static Value operator /(Value left, Value right) => left * right.Pow(-1);

    
    public Value Neg()
    {
        return this * -1;
    }
    
    public Value Pow(float other)
    {
        var x = Data;
        var t = Math.Pow(x, other);

        var o = new Value((float)t, [this], "pow");
        o._backward = () =>
        {
            Grad += (float)(other * Math.Pow(x, other - 1)) * o.Grad;
            return 0.0f;
        };
        return o;
    }

    public void Backward()
    {
        var topo = new List<Value>();
        var visited = new HashSet<Value>();

        void BuildTopo(Value v)
        {
            if (!visited.Contains(v))
            {
                visited.Add(v);
                foreach (var child in v.Previous)
                {
                    BuildTopo(child);
                }
                topo.Add(v);
            }
        }
        BuildTopo(this);
        this.Grad = 1.0f;
        topo.Reverse();
        foreach (var node in topo) node._backward?.Invoke();
    }
    


    public Value TanH()
    {
        var x = Data;
        var t = (Math.Exp(2*x) -1) / (Math.Exp(2*x) + 1);
        
        var o = new Value((float)t, [this], "tanh");
        o._backward = () =>
        {
            Grad += (float)((1 - Math.Pow(t,2.0)) * o.Grad);
            return 0.0f;
        };
        return o;
    }

    public Value Relu()
    {
        var x = (Data < 0.0f ? 0 : Data);
        var o = new Value(x, "ReLU");

        o._backward = () =>
        {
            Grad += (o.Grad > 0 ? 1.0f : 0.0f) * o.Grad;
            return 0.0f;
        };
        
        return o;
    }

    public Value Exp()
    {
        var x = Data;

        var o = new Value((float)Math.Exp(Data), [this], "exp");
        o._backward = () =>
        {
            Grad += (o.Data * o.Grad);
            return 0.0f;
        };
        return o;
    }

    
}
