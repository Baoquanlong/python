from typing import NewType


class Screen:
    '''
        请利用@property给一个Screen对象加上width和height属性，以及一个只读属性resolution
    '''
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self,value):
        if not isinstance(value,int):
            raise ValueError('wrong type,need integer')
        if value<0 :
            raise ValueError('value mast be positive number')
        self.width = value
        
    
    @property
    def height(self):
        return self._heigth
    @height.setter
    def height(self,value):
        if not isinstance(value,int):
            raise ValueError('wrong type,need integer')
        if value<0 :
            raise ValueError('value mast be positive number')
        self.height = value



def main():
    s = Screen()
    s.width =10
    s.height = 20
    print(s.height)
    print(s.width)


main()