
/*

GNU GPL V3 License

Copyright (c) 2020 Eric Tondelli. All rights reserved.

This file is part of Ai4prod.

Ai4prod is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Ai4prod is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Ai4prod.  If not, see <http://www.gnu.org/licenses/>

*/

#pragma once

namespace ai4prod
{

    inline int CocoMap[80]={1,2,3,4,5,6,7,8,
                     9,10,11,13,14,15,16,17,
                     18,19,20,21,22,23,24,25,
                     27,28,31,32,33,34,35,36,
                     37,38,39,40,41,42,43,44,
                     46,47,48,49,50,51,52,53,
                     54,55,56,57,58,59,60,61,
                     62,63,64,65,67,70,72,73,
                     74,75,76,77,78,79,80,81,
                     82,84,85,86,87,88,89,90};

    struct InstanceSegmentationResult
    {

        torch::Tensor boxes;
        torch::Tensor masks;
        torch::Tensor classes;
        torch::Tensor scores;
        torch::Tensor proto;
    };

} // namespace ai4prod
