
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
#include <iostream>
#include "torch/torch.h"
#include "explain_export.h"
#pragma once


namespace ai4prod
{

    namespace explain
    {

        class EXPLAIN_EXPORT ConfusionMatrix
        {
        private:

            void parseCsv();
            std::string m_sPathToCsv;
            std::vector<std::vector<float>> m_SVConfusionMatrix;
        public:
            
            ConfusionMatrix();
            bool init(std::string pathToCF,int numOfClasses);

            torch::Tensor getProbability(torch::Tensor result);

            std::vector<std::vector<float>> getConfutionMatrix(){
                return m_SVConfusionMatrix;
            }
            
            
            ~ConfusionMatrix();
        };

    }

}