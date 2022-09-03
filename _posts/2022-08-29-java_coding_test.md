---
layout: single
title: "자바 코테 기본기"
categories : java
tag: [Algorithm, java]
toc: true
author_profile : false
use_math: true
sidebar:
     nav: "docs"


---



## 기초

### 출력

> 단어출력
>
> ```java
> 
> public class Main {
>     public static void main (String args[]) {
>         System.out.print("World");
>     }
> }
> ```
>
> 



### 입력

>정수입력

>```java
>import java.util.Scanner;
>
>public class Main {
>    public static void main (String args[]) {
>        Scanner sc = new Scanner(System.in);
>        int a;
>        a = sc.nextInt();
>        System.out.println(a);
>    }
>}
>
>```



>실수입력
>
>```java
> import java.util.Scanner;
>
>public class Main {
>    public static void main (String args[]) {
>        Scanner sc = new Scanner(System.in);
>        double a = sc.nextDouble();
>        System.out.println(a + 0.58);
>    }
>}
>
>```



> 문자입력
>
> ```java
> import java.util.Scanner;
> 
> public class Main {
>     public static void main (String args[]) {
>         Scanner sc = new Scanner(System.in);
> 
>         // 변수 선언
>         char c;
> 
>         // 입력
>         c = sc.next().charAt(0);      // 문자 입력
> 
>         // 출력
>         System.out.println(c);
>     }
> }
> ```
>
> 문자열입력
>
> ```java
> import java.util.Scanner;
> 
> public class Main {
>     public static void main (String args[]) {
>         Scanner sc = new Scanner(System.in);
>         String s = sc.next();
>         System.out.println(s);
>     }
> }
> ```
>
> 구분자
>
> ```java
> import java.util.Scanner;
> 
> public class Main {
>     public static void main (String args[]) {
>         // 변수 선언 및 입력
>         Scanner sc = new Scanner(System.in);
>         sc.useDelimiter(":");
>         int h = sc.nextInt();
>         int m = sc.nextInt();
>         
>         // 출력
>         System.out.println((h + 1) + ":" + m);
>     }
> }
> ```
>
> 

### 배열

>```java
>import java.util.Scanner;
>
>public class Main {
>    public static void main(String[] args) {
>        Scanner sc = new Scanner(System.in);
>
>        int[] arr = new int[10];
>        int val, sum;
>        sum = 0;
>        for (int i = 0; i < 10; i++) {
>            arr[i] = sc.nextInt();
>            sum += arr[i];
>        }
>        System.out.print(sum);
>    }
>}
>
>```



### 최대최소

> ```java
> public class Main {
>     public static void main(String[] args) {
>         final int INT_MIN = Integer.MIN_VALUE;
> 
>         int[] arr = new int[]{ -1, -5, -2, -5, -3, -9 };
>         int maxVal = INT_MIN;
>         for (int i = 0; i < 6; i++) {
>             if (arr[i] > maxVal) {
>                 maxVal = arr[i];
>             }
>         }
> 
>         System.out.println(maxVal);
>     }
> }
> 
> ```



### 2차원배열입력

```java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int[][] arr2d = new int[4][4];

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                arr2d[i][j] = sc.nextInt();
            }
        }

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                System.out.print(arr2d[i][j] + " ");
            }
            System.out.println();
        }
    }
}


입력
1 2 3 4
7 8 9 10
11 12 13 14
15 16 17 18

출력
1 2 3 4
7 8 9 10
11 12 13 14
15 16 17 18
```



### 문자열입력

```java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String str = sc.next();
        System.out.println(str);
    }
}

// 입력
>> hello world

// 출력
hello
  
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String str = sc.nextLine();
        System.out.println(str);
    }
}

// 입력
>> hello world

// 출력
hello world

```



### 문자열 리스트 변환

```java
public class Main {
    public static void main(String[] args) {
        String s = "baaana";
        char[] arr = s.toCharArray();

        arr[2] = 'n';
        s = String.valueOf(arr);

        System.out.println(s);
    }
}
```



### 문자열에서 문자제거

```java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // 문자열을 입력받습니다.
        String str = sc.next();

        // 문자열의 길이를 구합니다.
        int len = str.length();

        // 앞에서 2번째 원소를 제거합니다. (이때 문자열의 길이가 1 감소하는것을 반드시 기억합니다)
        str = str.substring(0, 1) + str.substring(2);
        len--;

        // 뒤에서 2번째 원소를 제거합니다.
        str = str.substring(0, len - 2) + str.substring(len - 1);
        len--;
            
        // 앞에서 2번째, 뒤에서 2번째 원소가 제거된 문자열을 출력합니다.
        System.out.println(str);
    }
}
```



### 문자열을 정수로 변환

```java
public class Main {
    public static void main(String[] args) {
        String a = "123";
        int aInt = Integer.parseInt(a) + 1;

        System.out.println(aInt);
    }
}

```



### 정수를 문자열로

```Java
public class Main {
    public static void main(String[] args) {
        int a = 123;
        String aStr;

        aStr = Integer.toString(a);
        System.out.println(aStr);
    }
}

>> 123

```



### 문자열비교

```java
public class Main {
    public static void main(String[] args) {
        String a = "abc";
        String b = "abc";
        String c = "cba";

        System.out.println(a.equals(b)); // true
        System.out.println(a.equals(c)); // false
    }
}

```



```java
public class Main {
    public static void main(String[] args) {
        String a = "abc";
        String b = "abd";
        String c = "aba";

        System.out.println(a.compareTo(b)); // -1
        System.out.println(a.compareTo(c)); // 2
        System.out.println(a.compareTo(a)); // 0
    }
}
```



### 함수

```java
public class Main {
    public static void print5Stars() {
        for(int i = 0; i < 5; i++)
            System.out.print("*");
        System.out.println();
    }

    public static void main(String[] args) {
        for(int i = 0; i < 4; i++)
            print5Stars(); 
    }
}

>> *****
   *****
   *****
   *****

```



### swap

```java
class IntWrapper {
    int value;

    public IntWrapper(int value) {
        this.value = value;
    }
}

public class Main {
    public static void swap(IntWrapper n, IntWrapper m) {
        int temp = n.value;
        n.value = m.value;
        m.value = temp;
    }

    public static void main(String[] args) {
        IntWrapper n = new IntWrapper(10);
        IntWrapper m = new IntWrapper(20);

        swap(n, m);

        System.out.println(n.value + " " + m.value); // 20 10
    }
}

```



### array -> call by reference (how to remove side effect)

```java
public class Main {
    public static void modify(int[] arr2) {  // arr2는 arr와 관련이 없다.
        arr2[0] = 10;
    }

    public static void main(String[] args) {
        int[] arr = new int[]{1, 2, 3, 4};
        modify(arr.clone());                 // 새로운 배열을 만들어 넘기기

        for(int i = 0; i < 4; i++)
            System.out.print(arr[i] + " ");
    }
}

>> 1 2 3 4 # 값에 변화가 없다

```



### string -> immutable

```java
import java.util.Scanner;

public class Main {
    public static String str;

    public static boolean palindrome(String s) {
        for(int i = 0; i < s.length(); i++)
            if(s.charAt(i) != s.charAt(s.length() - i - 1))
                return false;
        
        return true;
    }

    public static void main(String[] args) {
        // 변수 선언 및 입력:
        Scanner sc = new Scanner(System.in);
        str = sc.next();

        if(palindrome(str))
            System.out.print("Yes");
        else
            System.out.print("No");
    }
}
```



### 정렬

```java
import java.util.Arrays;
import java.util.Collections;

public class Main {
    public static void main(String[] args) {
        int[] arr = new int[]{12, 41, 37, 81, 19, 25, 60, 20}; 
        Integer[] arr2 = Arrays.stream(arr).boxed().toArray(Integer[]::new);
        Arrays.sort(arr2, Collections.reverseOrder());

        for(int i = 0; i < 8; i++) // 81, 60, 41, 37, 25, 20, 19, 12
            System.out.print(arr2[i] + " ");
    }
}

```



### class

```java
class Student {
    int k, e, m;

    public Student(int kor, int eng, int math){
        this.k = kor;
        this.e = eng;
        this.m = math;
    }
};

public class Main {
    public static void main(String[] args) {
        Student student1 = new Student(90, 80, 90);

        System.out.println(student1.k);  // 90
        System.out.println(student1.e);  // 80
        System.out.println(student1.m); // 90
    }
}

```



> constructor
>
> ```java
> class Student {
>     int kor, eng, math;
> 
>     public Student(){
>         this.kor = 0;
>         this.eng = 0;
>         this.math = 0;
>     }
> 
>     public Student(int kor, int eng, int math){
>         this.kor = kor;
>         this.eng = eng;
>         this.math = math;
>     }
> };
> 
> public class Main {
>     public static void main(String[] args) {
>         Student student2 = new Student();  // 값이 넘어가지 않는 생성자를 이용
>         System.out.println(student2.kor);  // 0
>         System.out.println(student2.eng);  // 0
>         System.out.println(student2.math); // 0
> 
>         student2.kor = 90;
>         student2.eng = 80;
>         student2.math = 90;
> 
>         System.out.println(student2.kor);  // 90
>         System.out.println(student2.eng);  // 80
>         System.out.println(student2.math); // 90
>     }
> }
> 
> ```



> instance sort
>
> ```java
> import java.util.Arrays;
> 
> class Student{
>     int kor, eng, math;
> 
>     public Student(int kor, int eng, int math){
>         this.kor = kor;
>         this.eng = eng;
>         this.math = math;
>     }
> };
> 
> public class Main {
>     public static void main(String[] args) {
>         Student[] students = new Student[] {
>             new Student(90, 80, 90), // 첫 번째 학생
>             new Student(20, 80, 80), // 두 번째 학생
>             new Student(90, 30, 60), // 세 번째 학생
>             new Student(60, 10, 50), // 네 번째 학생
>             new Student(80, 20, 10)  // 다섯 번째 학생 
>         };
> 
>         Arrays.sort(students, (a, b) -> a.kor - b.kor); // 국어 점수 기준 오름차순 정렬
> 
>         for(int i = 0; i < 5; i++)
>             System.out.println(students[i].kor + " " + students[i].eng + " " + students[i].math);
>     }
> }
> 
> 
> ```
>
> ```java
> import java.util.Arrays;
> 
> class Student implements Comparable<Student> {
>     int kor, eng, math;
> 
>     public Student(int kor, int eng, int math){
>         this.kor = kor;
>         this.eng = eng;
>         this.math = math;
>     }
> 
>     @Override
>     public int compareTo(Student student) {
>         if(this.kor == student.kor)         // 국어 점수가 일치한다면
>             return student.eng - this.eng;  // 영어 점수를 기준으로 내림차순 정렬합니다.
>         return this.kor - student.kor;	     // 국어 점수가 다르다면, 오름차순 정렬합니다.
>     }
> };
> 
> public class Main {
>     public static void main(String[] args) {
>         Student[] students = new Student[] {
>             new Student(90, 80, 90), // 첫 번째 학생
>             new Student(20, 80, 80), // 두 번째 학생
>             new Student(90, 30, 60), // 세 번째 학생
>             new Student(60, 10, 50), // 네 번째 학생
>             new Student(80, 20, 10)  // 다섯 번째 학생 
>         };
> 
>         Arrays.sort(students);
> 
>         for(int i = 0; i < 5; i++)
>             System.out.println(students[i].kor + " " + students[i].eng + " " + students[i].math);
>     }
> }
> 
> >> 20 80 80
>    60 10 50
>    80 20 10
>    90 80 90
>    90 30 60
> 
> ```
>
> ```java
> import java.util.Scanner;
> import java.util.Arrays;
> import java.util.Comparator;
> 
> // 학생들의 정보를 나타내는 클래스 선언
> class Student {
>     String name;
>     int height;
>     double weight;
> 
>     public Student(String name, int height, double weight){
>         this.name = name;
>         this.height = height;
>         this.weight = weight;
>     }
> };
> 
> public class Main {
>     public static void main(String[] args) {
>         Scanner sc = new Scanner(System.in);
> 
>         // 변수 선언 및 입력:
>         int n = 5;
>         Student[] students = new Student[n];
>         for(int i = 0; i < n; i++) {
>             String name = sc.next();
>             int height = sc.nextInt();
>             double weight = sc.nextDouble();
> 
>             // Student 객체를 생성해 리스트에 추가합니다.
>             students[i] = new Student(name, height, weight);
>         }
> 
>         // custom comparator를 활용한 정렬
>         Arrays.sort(students, new Comparator<Student>() {  
>             @Override
>             public int compare(Student a, Student b) { // 이름 기준 오름차순 정렬합니다.
>                 return a.name.compareTo(b.name);
>             }
>         });
> 
>         // 이름순으로 정렬한 결과를 출력합니다.
>         System.out.println("name");
>         for (int i = 0; i < n; i++){
>             System.out.print(students[i].name + " ");
>             System.out.print(students[i].height + " ");
>             System.out.printf("%.1f\n", students[i].weight);
>         }
> 
>         System.out.println();
> 
>          // custom comparator를 활용한 정렬
>         Arrays.sort(students, new Comparator<Student>() {  
>             @Override
>             public int compare(Student a, Student b) { // 키 기준 내림차순 정렬합니다.
>                 return b.height - a.height;
>             }
>         });
> 
>         // 키순으로 정렬한 결과를 출력합니다.
>         System.out.println("height");
>         for (int i = 0; i < n; i++){
>             System.out.print(students[i].name + " ");
>             System.out.print(students[i].height + " ");
>             System.out.printf("%.1f\n", students[i].weight);
>         }
>     }
> }
> ```
>
> 