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

#include "explain.h"

namespace ai4prod
{

    namespace explain
    {

        ConfusionMatrix::ConfusionMatrix()
        {
        }

        void ConfusionMatrix::parseCsv()
        {

            std::ifstream data(m_sPathToCsv);
            std::string line;
            std::vector<std::vector<std::string>> parse;
            while (std::getline(data, line))
            {
                std::stringstream lineStream(line);
                std::string cell;
                std::vector<std::string> parsedRow;
                while (std::getline(lineStream, cell, ','))
                {
                    parsedRow.push_back(cell);
                }

                parse.push_back(parsedRow);
            }

            //convert String to Float
            m_SVConfusionMatrix.reserve(parse.size());
            for (auto &&v : parse)
            {
                std::vector<float> tmp ;

                std::transform(v.begin(), v.end(), back_inserter(tmp), [](const std::string & astr){ return std::stod( astr) ; } ) ;
                m_SVConfusionMatrix.emplace_back(std::begin(tmp), std::end(tmp));
            }
        }
        bool ConfusionMatrix::init(std::string pathToCF, int numOfClasses)
        {
            m_sPathToCsv = pathToCF;

            //load confusion Matrix from Csv
            parseCsv();

            
            return true;

            //load confusion Matrix in memory
        }

        

        ConfusionMatrix::~ConfusionMatrix()
        {
        }

    }

}
