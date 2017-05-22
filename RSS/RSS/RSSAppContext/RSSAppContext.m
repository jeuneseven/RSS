//
//  RSSAppContext.m
//  RSS
//
//  Created by 李占昆 on 17/5/23.
//  Copyright © 2017年 kkk. All rights reserved.
//

#import "RSSAppContext.h"

@implementation RSSAppContext

+ (RSSAppContext*)sharedAppContext
{
    static dispatch_once_t pred = 0;
    __strong static id sharedObject = nil;
    dispatch_once(&pred, ^{
        sharedObject = [[self alloc] init];
    });
    
    return sharedObject;
}

@end
