#ifndef __MINMAX__
#define __MINMAX__

class MinMax
{
  bool b_ = true;
  double xMin_, xMax_;

public:
    MinMax() {}

  double min() const {return xMin_;}
  double max() const {return xMax_;}

  
  void update(double x)
  {
    if (b_)
      {
        xMin_ = xMax_ = x;
	b_ = false;
        return;
      }
    if (x < xMin_)
      {
        xMin_ = x;
      }
    else if (x > xMax_)
      {
        xMax_ = x;
      }
  }

};

#endif
