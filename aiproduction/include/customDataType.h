
namespace ai4prod
{

    struct YolactResult
    {

        torch::Tensor boxes;
        torch::Tensor masks;
        torch::Tensor classes;
        torch::Tensor scores;
    };

} // namespace ai4prod